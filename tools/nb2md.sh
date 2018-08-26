#!/bin/bash
for src in notebook/binary/*.ipynb
do
    jupyter nbconvert --to markdown --output-dir _includes/notebook/binary $src
done
