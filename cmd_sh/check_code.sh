#!/bin/bash
#####################################
## File name : check_code.sh
## Create date : 2018-11-25 15:57
## Modified date : 2019-02-13 20:10
## Author : DARREN
## Describe : not set
## Email : lzygzh@126.com
####################################

realpath=$(readlink -f "$0")
export basedir=$(dirname "$realpath")
export filename=$(basename "$realpath")
export PATH=$PATH:$basedir/dlbase
export PATH=$PATH:$basedir/dlproc
#base sh file
. dlbase.sh
#function sh file
. etc.sh

source $env_path/py2env/bin/activate
pylint --rcfile=pylint.conf base_graph.py
pylint --rcfile=pylint.conf etc.py
pylint --rcfile=pylint.conf mnist_data_set.py
pylint --rcfile=pylint.conf show.py
pylint --rcfile=pylint.conf stn_model.py
pylint --rcfile=pylint.conf graph.py
pylint --rcfile=pylint.conf record.py
pylint --rcfile=pylint.conf test_graph.py
pylint --rcfile=pylint.conf main.py
pylint --rcfile=pylint.conf status.py
pylint --rcfile=pylint.conf train_graph.py

pip freeze > python3_requiements.txt
deactivate

source $env_path/py3env/bin/activate
pylint --rcfile=pylint.conf base_graph.py
pylint --rcfile=pylint.conf etc.py
pylint --rcfile=pylint.conf mnist_data_set.py
pylint --rcfile=pylint.conf show.py
pylint --rcfile=pylint.conf stn_model.py
pylint --rcfile=pylint.conf graph.py
pylint --rcfile=pylint.conf record.py
pylint --rcfile=pylint.conf test_graph.py
pylint --rcfile=pylint.conf main.py
pylint --rcfile=pylint.conf status.py
pylint --rcfile=pylint.conf train_graph.py
pip freeze > python2_requiements.txt
deactivate
