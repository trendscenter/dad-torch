mode="dad"
nsites=4
real_nsites=3
for i in $(seq 0 $real_nsites);
do echo $i;
  if [ $i -eq 0 ]
    then
    echo master! $i
    sbatch -J $mode-$i -o logs/master-$mode-$i.log -e logs/master-$mode-$i.err run_master.sh 0 $mode $nsites
  else
    echo slave! $i
    sbatch -J $mode-$i -o logs/slave-$mode-$i.log -e logs/slave-$mode-$i.err run_slave.sh $i $mode $nsites
  fi
done