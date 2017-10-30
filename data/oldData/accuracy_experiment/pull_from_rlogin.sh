#!/bin/bash
for i in `seq 1 10`;
do
          scp ericrw96@rlogin.cs.vt.edu:out.all$i .
done

