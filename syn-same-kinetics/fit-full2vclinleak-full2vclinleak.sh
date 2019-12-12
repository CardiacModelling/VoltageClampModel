#!/usr/bin/env bash

cd ..; source env/bin/activate; cd -

## Make sure logging folders exist
LOG='./log/fit-syn-full2vclinleak-full2vclinleak'
mkdir -p $LOG
echo '## fit syn.' >> log/save_pid.log

## for cell in $CELLS  # or
for ((x=0; x<5; x++));
do
	echo "Cell $x"
	nohup python fit-full2vclinleak-full2vclinleak.py $x 10 > $LOG/cell_$x.log 2>&1 &
	echo "# cell $x" >> log/save_pid.log
	echo $! >> log/save_pid.log
	sleep 5
done

