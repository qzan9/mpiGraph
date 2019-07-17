#!/usr/bin/env bash

set -e

if [[ ${#} -eq 1 ]]; then
	NP=${1}
	HOSTFILE=${PWD}/hostfile
elif [[ ${#} -eq 2 ]]; then
	NP=${1}
	HOSTFILE=${2}
else
	echo "Usage: $(basename ${0}) [num of proc] [hostfile]"
	exit 1
fi

if [[ ${NP} -gt 30000 ]]; then
	NPC=$(($NP/4))
	/usr/bin/ulimit -u 65536 -n 65536
else
	NPC=${NP}
fi

UCX_PATH=/opt/hpc/software/mpi/hpcx/v2.4.0_ga/ucx/rocm-mt
OMPI_PATH=/opt/hpc/software/mpi/hpcx/v2.4.0_ga/ompi-rocm
export OPAL_PREFIX=${OMPI_PATH}
export LD_LIBRARY_PATH=${OMPI_PATH}/lib

echo -en "\033[1;32mMPIGRAPH-GDR BEGINS: "; date; echo -en "\033[0m"
${UCX_PATH}/bin/ucx_info -v
${OMPI_PATH}/bin/ompi_info --version
${OMPI_PATH}/bin/mpirun\
	--allow-run-as-root\
	--np ${NP} --hostfile ${HOSTFILE} --map-by ppr:4:node --bind-to none\
	--mca plm_rsh_no_tree_spawn 1\
		--mca plm_rsh_num_concurrent ${NPC} --mca routed_radix ${NPC}\
	${PWD}/mpigraph_module\
		2>&1 | /usr/bin/tee ${PWD}/mpigraph_log-$(date +%y%m%d_%H%M%S)
echo -en "\033[1;32mMPIGRAPH-GDR ENDS: "; date; echo -en "\033[0m"
