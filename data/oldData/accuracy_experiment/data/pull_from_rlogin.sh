#!/bin/bash
for i in `seq 1 10`;
do
          scp ericrw96@rlogin.cs.vt.edu:train_data$i.rw .
	  scp ericrw96@rlogin.cs.vt.edu:test_data$i.rw .
done

