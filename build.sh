#!/bin/sh
cd encoder
make
mv LayerNormPlugin.so ../
python 03-ModifyModel.py
python encoder-surgeonLayerNorm.py
python encoder-ToTrT.py
mv encoder.plan ../

cd ../decoder
python decoder-surgeonLayerNorm.py
python decoder-ToTrT.py
mv decoder.plan ../
