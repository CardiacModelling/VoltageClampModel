#!/usr/bin/env bash

## Make sure logging folders exist
LOG='./log/fit-syn-simvclinleak-simvclinleak'
mkdir -p $LOG
echo '## fit syn.' >> log/save_pid.log

## for cell in $CELLS  # or
for ((x=0; x<50; x++));
do
	echo "Cell $x"
	nohup python fit-simvclinleak-simvclinleak.py $x 10 > $LOG/cell_$x.log 2>&1 &
	echo "# cell $x" >> log/save_pid.log
	echo $! >> log/save_pid.log
	sleep 5
done

