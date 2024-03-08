# End-to-end ML pipeline
 
This is a simple git repo where we build an end to end Machine Learning pipeline.
It contains the following features:
1) Github Actions
2) MLFlow
3) Dagshub
4) DVC



### MLFlow

### DagsHub

In order to see the results of the experiments also in DagsHub, you need to do the following every time you open the terminal.

Export:
* MLFLOW_TRACKING_URI
* MLFLOW_TRACKING_USERNAME
* MLFLOW_TRACKING_PASSWORD 

from DagsHub after connecting the repository. In the DagsHub page for your repo, you can find this information
by clicking the green "Remote" button.

Run the following shell commands

```bash
export MLFLOW_TRACKING_URI="insert here the value"
export MLFLOW_TRACKING_USERNAME="insert here the value"
export MLFLOW_TRACKING_PASSWORD="insert here the value"
```

### DVC

Initialize DVC using

```bash
dvc init
```

which generates a .dvcignore file and a .dvc folder.
Then, you can run the entire pipeline by using the command

```bash
dvc repro
```
which runs the dvc.yaml file where we defined the stages of the pipeline
with their dependencies and parameters.

The pipeline will not run again stages where there haven't been any changes, which
can save a lot of time.

Lastly, you can use
```bash
dvc dag
```
to visualize the graph showing the dependencies between the different
components of the pipeline.
