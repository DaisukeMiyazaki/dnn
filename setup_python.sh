#!/bin/zsh
pipenv shell
python --version
pip list

# start ruuning jupyter-notebook
jupyter-notebook

# カーネル名の一覧を表示する
jupyter kernelspec list 

# vscode select kernelから出てきた環境を追加する
# ex.) /Users/daisuke/.local/share/virtualenvs/nn-bsgCLbAH/share/jupyter/kernels/python3