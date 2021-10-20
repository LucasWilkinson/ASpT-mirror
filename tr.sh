pp=$1
dataset=$2
dimens=$3

echo "val dd = loadSMat(\"${pp}/smat/${dataset}.ntx.smat.lz4\");" > alp.ssc

echo "val (nn,opts)=LDAgibbs.learner(dd)" >> alp.ssc

echo "opts.dim=$dimens;">> alp.ssc
echo "opts.uiter=1;">> alp.ssc
echo "opts.batchSize=400000;">> alp.ssc
echo "opts.npasses=8;">> alp.ssc
echo "opts.useBino=true;">> alp.ssc
echo "opts.doDirichlet=true;">> alp.ssc
echo "opts.alpha=0.2f;">> alp.ssc
echo "opts.doAlpha=true;">> alp.ssc
echo "opts.nsamps=100;">> alp.ssc
echo "opts.power=0.5f;">> alp.ssc

echo "nn.train">> alp.ssc
echo "sys.exit">> alp.ssc

# rm alp.ssc
