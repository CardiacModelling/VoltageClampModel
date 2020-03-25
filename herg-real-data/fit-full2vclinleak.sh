#!/usr/bin/env bash

FILENAME="herg25oc1"

# Make sure logging folders exist
LOG="./log/fit-full2vclinleak-$FILENAME"
mkdir -p $LOG

# (.) turns grep return into array
# use grep with option -e (regexp) to remove '#' starting comments
CELLS=(`grep -v -e '^#.*' ../manualselection/manualv2selected-$FILENAME.txt`)

echo "## fit-$FILENAME" >> log/save_pid.log

# Load up virtual env
cd ..; source env/bin/activate; cd -

# for cell in $CELLS  # or
for ((x=0; x<3; x++));
do
	echo "${CELLS[x]}"
	nohup python fit-full2vclinleak.py $FILENAME ${CELLS[x]} 10 > $LOG/${CELLS[x]}.log 2>&1 &
	echo "# ${CELLS[x]}" >> log/save_pid.log
	echo $! >> log/save_pid.log
	sleep 2
done

