sudo apt-get update && sudo apt-get install -y tmux
tmux new qwenrun
sudo apt-get install -y linux-tools-common linux-tools-generic
sudo apt-get install -y linux-tools-$(uname -r)
cpupower frequency-info