projects_path="/Users/me/Documents/rtcloud-projects"

#project server
docker run --name projectServer --publish 8888:8888 \
           -it --rm --volume $projects_path:/rt-cloud/projects \
           brainiak/rtcloud:latest scripts/data_analyser.sh \
           --projectName induction --subjectRemote --test

#analysis listner
docker run --name analysisListener -it --rm \
           --volume $projects_path/induction/outDir:/rt-cloud/outDir \
           brainiak/rtcloud:latest scripts/analysis_listener.sh \
           --server 172.17.0.1:8888 --username test --password test --test