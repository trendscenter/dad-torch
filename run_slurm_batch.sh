mode=$1
nsites=$2
real_nsites=$((nsites - 1))
project=$3
for i in $(seq 0 $real_nsites); 
do echo $i; 
if [ $i -eq 0 ]
then
echo master! $i
Submit_Output="$(sbatch -J $mode-$i -o logs/${project}-master-$mode-$i.log -e logs/${project}-master-$mode-$i.err run_master.sh 0 $mode $nsites $project >&1)"
echo Submit_Output $Submit_Output
JobId=`echo $Submit_Output | grep 'Submitted batch job' | awk '{print $4}'`
echo JobId $JobId
sleep 60
Host=`scontrol show job ${JobId} | grep ' NodeList' | awk -F'=' '{print $2}' | nslookup | grep 'Address: ' | awk -F': ' '{print $2}'`
echo host $Host
else
echo slave! $i
sbatch -J $mode-$i -o logs/${project}-slave-$mode-$i.log -e logs/${project}-slave-$mode-$i.err run_slave.sh $i $mode $nsites $project $Host
fi
done
