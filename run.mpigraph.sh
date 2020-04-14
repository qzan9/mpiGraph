#!/usr/bin/env bash

set -e

TIMESTAMP=$(date +%y%m%d_%H%M%S)

if [[ -n ${1} ]]; then
  HOSTFILE=${1}
else
  HOSTFILE=${PWD}/hostfile
fi
NNODE=$(wc -l ${HOSTFILE} | cut -f1 -d' ')
# root privilege required
#if [[ ${NNODE} -ge 256 ]]; then
#  ulimit -u 65536 -n 65536
#fi
if [[ ${NNODE} -ge 8000 ]]; then
  NPD=4
else
  NPD=1
fi
NNODESTR=nnode$(printf "%05d" ${NNODE})
if [[ ${NNODE} -eq 20 ]]; then
  NNODESTR=${NNODESTR}-$(head -n1 ${HOSTFILE} | cut -c1-5)
fi

PERF_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
clusconf -f ${HOSTFILE} --sync-file ${PERF_DIR}/mpiGraph\
                                    ${PERF_DIR}/mpigraph_module; echo
UCX_PATH=/opt/hpc/software/mpi/ucx_without_rocm
OMPI_PATH=/opt/hpc/software/mpi/hpcx/v2.4.0/ompi
export OPAL_PREFIX=${OMPI_PATH}
export LD_LIBRARY_PATH=${UCX_PATH}/lib:${OMPI_PATH}/lib:${LD_LIBRARY_PATH}
UCX_INFO=${UCX_PATH}/bin/ucx_info
OMPI_INFO=${OMPI_PATH}/bin/ompi_info
MPIRUN=${OMPI_PATH}/bin/mpirun

pushd ${PERF_DIR} > /dev/null

SHA256=$(sha256sum ${PWD}/mpiGraph | cut -c1-7)

LOG_DIR=${PWD}/log
[ -d ${LOG_DIR} ] || mkdir -p ${LOG_DIR}
LOG=${LOG_DIR}/mpigraph_${SHA256}-${NNODESTR}-${TIMESTAMP}

echo -e "\033[1;31mMPIGRAPH TEST BEGINS: $(date)\033[0m\n"
${UCX_INFO} -v 2>&1 | sed '$a\\' | tee -a ${LOG}
${OMPI_INFO} -a | sed -n '2,3p' | sed "s/^[ \t]*/# /" | tee -a ${LOG}
${OMPI_INFO} -a | grep "Configure command line" | sed "s/^[ \t]*/# /" | sed '$a\\' | tee -a ${LOG}
for PPR in 1 4; do
  NP=$(($PPR*$NNODE)); NPC=$(($NP/$NPD))
  printf "\033[1;32mNP_PER_NODE=[${PPR}], NNODE=[%5d] - $(date)\033[0m\n" ${NNODE} | tee -a ${LOG}
  ${MPIRUN}\
    --allow-run-as-root\
    --np ${NP} --hostfile ${HOSTFILE} --map-by ppr:${PPR}:node --bind-to none\
    --mca plm_rsh_no_tree_spawn 1\
      --mca plm_rsh_num_concurrent ${NPC} --mca routed_radix ${NPC}\
    ${PWD}/mpigraph_module 2>&1 | tee -a ${LOG}
  echo | tee -a ${LOG}
done
echo -e "\n\033[1;31mMPIGRAPH TEST ENDS: $(date)\033[0m\n"

popd > /dev/null
