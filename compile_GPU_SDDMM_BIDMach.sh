#!/bin/bash

rm -Rf BIDMach
git clone http://github.com/BIDData/BIDMach
cp Learner.scala ./BIDMach/src/main/scala/BIDMach
cd BIDMach
mvn package
cd ..

cd BIDMach
mvn package 
cd ..
