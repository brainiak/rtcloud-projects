if [[ "$OSTYPE" == "darwin"* ]]; then
    cd ~/Documents/rt-cloud
    bash scripts/data_analyser.sh --projectDir ~/Documents/rt-press/real-time -p rt-press --test
else
    cd ~/software/rt-cloud
    bash scripts/data_analyser.sh --projectDir ~/rt-press/real-time -p rt-press --port 8073 --subjectRemote --test
fi

#this is for the remote analysis listner

docker run --name projectServer --publish 8888:8888 \
           -it --rm --volume /Users/ah7700/Documents/fc-demo:/rt-cloud/projects \
           brainiak/rtcloud:latest scripts/data_analyser.sh \
           --projectName functionalConnectivity --subjectRemote --test

docker run --name analysisListener -it --rm \
           --volume /Users/ah7700/Documents/fc-demo/functionalConnectivity/outDir:/rt-cloud/outDir \
           brainiak/rtcloud:latest scripts/analysis_listener.sh \
           --server 172.17.0.1:8888 --username test --password test --test