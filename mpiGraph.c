/*
Copyright (c) 2007-2008, Lawrence Livermore National Security (LLNS), LLC
Produced at the Lawrence Livermore National Laboratory (LLNL)
Written by Adam Moody <moody20@llnl.gov>.
UCRL-CODE-232117.
All rights reserved.

This file is part of mpiGraph. For details, see
  http://www.sourceforge.net/projects/mpigraph
Please also read the Additional BSD Notice below.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the disclaimer below.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the disclaimer (as noted below) in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the LLNL nor the names of its contributors may be used to
  endorse or promote products derived from this software without specific prior
  written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL LLNL, THE U.S. DEPARTMENT OF ENERGY OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.

Additional BSD Notice
1. This notice is required to be provided under our contract with the U.S.
Department of Energy (DOE). This work was produced at LLNL under Contract No.
W-7405-ENG-48 with the DOE.
2. Neither the United States Government nor LLNL nor any of their employees,
makes any warranty, express or implied, or assumes any liability or
responsibility for the accuracy, completeness, or usefulness of any information,
apparatus, product, or process disclosed, or represents that its use would not
infringe privately-owned rights.
3. Also, reference herein to any specific commercial products, process, or
services by trade name, trademark, manufacturer or otherwise does not
necessarily constitute or imply its endorsement, recommendation, or favoring by
the United States Government or LLNL. The views and opinions of authors
expressed herein do not necessarily state or reflect those of the United States
Government or LLNL and shall not be used for advertising or product endorsement
purposes.
*/

/* =============================================================
 * OVERVIEW: mpiGraph
 * Typically, one MPI task is run per node (or per interconnect link).  For a
 * job of N MPI tasks, the N tasks are logically arranged in a ring counting
 * ranks from 0 and increasing to the right, at the end rank 0 is to the right
 * of rank N-1.  Then a series of N-1 steps are executed.  In each step, each
 * MPI task sends to the task D units to the right and simultaneously receives
 * from the task D units to the left. The value of D starts at 1 and runs to
 * N-1, so that by the end of the N-1 steps, each task has sent to and received
 * from every other task in the run, excluding itself. At the end of the run,
 * two NxN matrices of bandwidths are gathered and reported to stdout -- one for
 * send bandwidths and one for receive bandwidths.
 * =============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

char  hostname[256];
char *hostnames;

/* =============================================================
 * ROCM MACROS
 * =============================================================
 */
#ifdef _USE_ROCM_

char VERS[] = "1.5-rocm";

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <assert.h>
#include <hip/hip_runtime_api.h>
#define HIP_CHECK(stmt)                                               \
  do {                                                                \
    hipError_t hip_err;                                               \
    hip_err = (stmt);                                                 \
    if (hipSuccess != hip_err) {                                      \
      fprintf(stderr, "[%s:%d] HIP call '%s' failed with 0x%x: %s\n", \
              __FILE__, __LINE__, #stmt, hip_err,                     \
              hipGetErrorString(hip_err));                            \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
    assert(hipSuccess == hip_err);                                    \
  } while (0)

#else

char VERS[] = "1.5";

#endif

/* =============================================================
 * TIMER MACROS
 * These macros start/stop the timer and measure the difference
 * =============================================================
 */
#ifdef USE_GETTIMEOFDAY
/* use gettimeofday() for timers */

#include <sys/time.h>
#define __TIME_START__ (gettimeofday(&g_timeval__start, &g_timezone))
#define __TIME_END_SEND__ (gettimeofday(&g_timeval__end_send, &g_timezone))
#define __TIME_END_RECV__ (gettimeofday(&g_timeval__end_recv, &g_timezone))
#define __TIME_USECS_SEND__ (d_Time_Diff_Micros(g_timeval__start, g_timeval__end))
#define __TIME_USECS_RECV__ (d_Time_Diff_Micros(g_timeval__start, g_timeval__end_recv))
#define d_Time_Diff_Micros(timeval__start, timeval__end) \
  ((double) ((timeval__end.tv_sec - timeval__start.tv_sec) * 1000000 + \
             (timeval__end.tv_usec - timeval__start.tv_usec)))
#define d_Time_Micros(timeval) ((double) (timeval.tv_sec * 1000000 + timeval.tv_usec))
struct timeval  g_timeval__start, g_timeval__end_send, g_timeval__end_recv;
struct timezone g_timezone;

#else
/* use MPI_Wtime() for timers instead of gettimeofday() (recommended)
 * on some systems gettimeofday may reset backwards via some global clock,
 * which leads to incorrect timing data including negative time periods
 */

#define __TIME_START__ (g_timeval__start = MPI_Wtime())
#define __TIME_END_SEND__ (g_timeval__end_send = MPI_Wtime())
#define __TIME_END_RECV__ (g_timeval__end_recv = MPI_Wtime())
#define __TIME_USECS_SEND__ ((g_timeval__end_send - g_timeval__start) * 1000000.0)
#define __TIME_USECS_RECV__ ((g_timeval__end_recv - g_timeval__start) * 1000000.0)
double g_timeval__start, g_timeval__end_send, g_timeval__end_recv;

#endif /* of USE_GETTIMEOFDAY */

/* =============================================================
 * MAIN TIMING LOGIC
 * Uses a ring-based (aka. shift-based) algorithm.
 * 1) First, logically arrange the MPI tasks
 *    from left to right from rank 0 to rank N-1 in a circular array.
 * 2) Then, during each step, each task sends messages to the task D uints to
 *    the right and receives from task D units to the left. In each step, each
 *    tasks measures its send and receive bandwidths.
 * 3) There are N-1 such steps so that each task has sent to and received from
 *    every task.
 * =============================================================
 */
void graph(int mypid, int nproc, int size, int niter, int window) {
  /* arguments are:
   *   mypid  = rank of this process
   *   nproc  = number of ranks
   *   size   = message size in bytes
   *   niter  = number of iterations to measure bandwidth between task pairs
   *   window = number of outstanding sends and recvs to a single rank
   */
  int i, j, k, w;

  /* allocate buffers for all of the messages */
  char *send_message, *recv_message;
#ifdef _USE_ROCM_
  HIP_CHECK(hipInit(0));
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMalloc((void **) &send_message, (window * size)));
  HIP_CHECK(hipMalloc((void **) &recv_message, (window * size)));
#else
  send_message = (char *) malloc(window * size);
  recv_message = (char *) malloc(window * size);
#endif

  /* allocate buffers for MPI communication */
  MPI_Request *request_array = (MPI_Request *) malloc(sizeof(MPI_Request) * window * 2);
  double *sendtimes = (double *) malloc(sizeof(double) * niter * nproc);
  double *recvtimes = (double *) malloc(sizeof(double) * niter * nproc);
  int *message_tags = (int *) malloc(window * sizeof(int));
  for (i = 0; i < window; i++) { message_tags[i] = i; }

  /* start iterating over distance */
  int distance = 1;
  while (distance < nproc) {
    /* this test can run for a long time, so print progress to screen as we go */
    float progress = (float) distance / (float) nproc * 100.0;
    if (0 == mypid) {
      printf("[%5d] of %d (%4.1f%%)\n", distance, nproc, progress);
      fflush(stdout);
    }

    /* find tasks distance units to the right (send) and left (recv) */
    int sendpid = (mypid + distance + nproc) % nproc;
    int recvpid = (mypid - distance + nproc) % nproc;

    /* run through 'niter' iterations on a given ring */
    for (i = 0; i < (niter + 1); i++) {
      /* couple of synch's to make sure everyone is ready to go */
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);

      if (i > 0) __TIME_START__;
      k = -1;
      /* fire off a window of isends to my send partner distance steps to my right */
      for (w = 0; w < window; w++) {
        k = k + 1;
        MPI_Isend(&send_message[w * size], size, MPI_BYTE, sendpid, message_tags[w],
                  MPI_COMM_WORLD, &request_array[k]);
      }
      /* setup a window of irecvs from my partner who is distance steps to my left */
      for (w = 0; w < window; w++) {
        k = k + 1;
        MPI_Irecv(&recv_message[w * size], size, MPI_BYTE, recvpid, message_tags[w],
                  MPI_COMM_WORLD, &request_array[k]);
      }
      /* time sends and receives separately */
      int flag_sends = 0;
      int flag_recvs = 0;
      while (!flag_sends || !flag_recvs) {
        /* check whether the recvs are done */
        if (!flag_recvs) {
          MPI_Testall((k + 1) / 2, &request_array[(k + 1) / 2 - 1], &flag_recvs, MPI_STATUS_IGNORE);
          if (flag_recvs) { if (i > 0) __TIME_END_RECV__; }
        }

        /* check whether the sends are done */
        if (!flag_sends) {
          MPI_Testall((k + 1) / 2, &request_array[0], &flag_sends, MPI_STATUS_IGNORE);
          if (flag_sends) { if (i > 0) __TIME_END_SEND__; }
        }
      }
      if (i > 0) {
        sendtimes[sendpid * niter + i - 1] = __TIME_USECS_SEND__ / (double) w;
        recvtimes[recvpid * niter + i - 1] = __TIME_USECS_RECV__ / (double) w;
      }
    } /* end loop */
    /* bump up the distance for the next ring step */
    distance++;
  } /* end distance loop */

  /* for each node, compute sum of my bandwidths with that node */
  if (0 == mypid) {
    printf("Gathering results ...\n");
    fflush(stdout);
  }
  double *sendsum = (double *) malloc(sizeof(double) * nproc);
  double *recvsum = (double *) malloc(sizeof(double) * nproc);
  for (j = 0; j < nproc; j++) {
    sendsum[j] = 0.0;
    recvsum[j] = 0.0;
    if (j == mypid) continue;
    for (i = 0; i < niter; i++) {
      sendsum[j] += sendtimes[j * niter + i];
      recvsum[j] += recvtimes[j * niter + i];
    }
  }

  /* gather send bw sums to rank 0 */
  double *allsum;
  if (0 == mypid) {
    allsum = (double *) malloc(sizeof(double) * nproc * nproc);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gather(sendsum, nproc, MPI_DOUBLE, allsum, nproc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* rank 0 computes send stats and prints result */
  if (0 == mypid) {
    /* compute stats over all nodes */
    double sendmin = 10000000000000000.0;
    double sendsum = 0.0;
    double sendmax = 0.0;
    double MBsec   = ((double) (size));
    for (j = 0; j < nproc; j++) {
      for (k = 0; k < nproc; k++) {
        if (j == k) continue;
        double sendval = allsum[j * nproc + k];
        sendmin = (sendval < sendmin) ? sendval : sendmin;
        sendsum += sendval;
        sendmax = (sendval > sendmax) ? sendval : sendmax;
      }
    }

    /* print send stats */
    sendmin /= (double) niter;
    sendsum /= (double) (nproc) * (nproc - 1) * niter;
    sendmax /= (double) niter;
    printf("\n");
    printf("Send max: %8.3f MB/s\n", MBsec / sendmin);
    printf("Send avg: %8.3f MB/s\n", MBsec / sendsum);
    printf("Send min: %8.3f MB/s\n", MBsec / sendmax);

    /* print send bandwidth table */
    printf("\n");
    printf("%-19s\t", "Send");
    for (k = 0; k < nproc; k++) {
      printf("%s:%-5d\t", &hostnames[k * sizeof(hostname)], k);
    }
    printf("\n");
    for (j = 0; j < nproc; j++) {
      printf("%s:%-5d %4s\t", &hostnames[j * sizeof(hostname)], j, "to");
      for (k = 0; k < nproc; k++) {
        double val = allsum[j * nproc + k];
        if (val != 0.0) { val = MBsec * (double) niter / val; }
        printf("%-15.3f\t", val);
      }
      printf("\n");
    }
  }

  /* gather recv bw sums to rank 0 */
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gather(recvsum, nproc, MPI_DOUBLE, allsum, nproc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* rank 0 computes recv stats and prints result */
  if (0 == mypid) {
    /* compute stats over all nodes */
    double recvmin = 10000000000000000.0;
    double recvsum = 0.0;
    double recvmax = 0.0;
    double MBsec   = ((double) (size));
    for (j = 0; j < nproc; j++) {
      for (k = 0; k < nproc; k++) {
        if (j == k) continue;
        double recvval = allsum[j * nproc + k];
        recvmin = (recvval < recvmin) ? recvval : recvmin;
        recvsum += recvval;
        recvmax = (recvval > recvmax) ? recvval : recvmax;
      }
    }

    /* print receive stats */
    recvmin /= (double) niter;
    recvsum /= (double) (nproc) * (nproc - 1) * niter;
    recvmax /= (double) niter;
    printf("\n");
    printf("Recv max: %8.3f MB/s\n", MBsec / recvmin);
    printf("Recv avg: %8.3f MB/s\n", MBsec / recvsum);
    printf("Recv min: %8.3f MB/s\n", MBsec / recvmax);

    /* print receive bandwidth table */
    printf("\n");
    printf("%-19s\t", "Recv");
    for (k = 0; k < nproc; k++) {
      printf("%s:%-5d\t", &hostnames[k * sizeof(hostname)], k);
    }
    printf("\n");
    for (j = 0; j < nproc; j++) {
      printf("%s:%-5d %4s\t", &hostnames[j * sizeof(hostname)], j, "from");
      for (k = 0; k < nproc; k++) {
        double val = allsum[j * nproc + k];
        if (val != 0.0) { val = MBsec * (double) niter / val; }
        printf("%-15.3f\t", val);
      }
      printf("\n");
    }
  }

  /* free off memory */
  if (0 == mypid) free(allsum);
  free(hostnames);
  free(sendsum);
  free(recvsum);
  free(sendtimes);
  free(recvtimes);
  free(message_tags);
  free(request_array);
#ifdef _USE_ROCM_
  HIP_CHECK(hipFree(send_message));
  HIP_CHECK(hipFree(recv_message));
#else
  free(send_message);
  free(recv_message);
#endif

  return;
}

/* =============================================================
 * MAIN DRIVER
 * Inits MPI, reads command-line parameters, and kicks off testing
 * =============================================================
 */
int main(int argc, char **argv) {
  int rank, nproc, size, niter, window;

  /* start up */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  /* set job parameters, read values from command line if they're there */
  size   = 1048576 * 4;
  niter  = 129;
  window = 64;
  if (argc == 4) {
    size   = atoi(argv[1]);
    niter  = atoi(argv[2]);
    window = atoi(argv[3]);
  }

  /* collect hostnames of all the processes */
  gethostname(hostname, sizeof(hostname));
  hostnames = (char *) malloc(sizeof(hostname) * nproc);
  MPI_Gather(hostname, sizeof(hostname), MPI_CHAR, hostnames, sizeof(hostname),
             MPI_CHAR, 0, MPI_COMM_WORLD);

  /* print the header */
  if (0 == rank) {
    /* mark start of output */
    printf("START mpiGraph v%s\n", VERS);
    printf("Msg Size: %d Bytes\nIterations: %d\nWindow Size: %d\n", size, niter, window);
    printf("Procs: %d\n\n", nproc);
    fflush(stdout);
  }

  /* synchronize, then start the run */
  MPI_Barrier(MPI_COMM_WORLD);
  graph(rank, nproc, size, niter, window);
  MPI_Barrier(MPI_COMM_WORLD);

  /* mark end of output */
  if (0 == rank) printf("\nEND mpiGraph\n");

  /* shut down */
  MPI_Finalize();
  return 0;
}
