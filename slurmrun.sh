git_branch="master" # Pass "local" to install local project
mode="dad"
nsites=4
real_nsites=3
backend="gloo"

for i in $(seq 0 $real_nsites);
do echo $i;
  if [ $i -eq 0 ]
    then
    echo master! $i
    sbatch -J $mode-$i -o logs/master-$mode-$i.log -e logs/master-$mode-$i.err run_master.sh 0 $mode $nsites $git_branch $backend
  else
    echo slave! $i
    sbatch -J $mode-$i -o logs/slave-$mode-$i.log -e logs/slave-$mode-$i.err run_slave.sh $i $mode $nsites $git_branch $backend
  fi
done