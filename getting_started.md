
1.1 Login to Lightning Studio ([lightning.ai/signup](https://lightning.ai/sign-up), use university email for quick approval) and start a studio with a GPU. Make sure to connect your github account.

1.2 Connect to that studio from terminal.

```bash
$ ssh <key>@ssh.lightning.com
```

1.3 Clone this repository inside studio and open in VS Code.

```bash
⚡~ git clone https://github.com/robotics-action-group/hand-rl.git
```
```bash
ur_computer:~$ code --folder-uri=vscode-remote://ssh-remote+<key>@ssh.lightning.ai/teamspace/studios/this_studio/hand-rl
```

1.4 Optionally: you could add this func to your `.bashrc`.
```bash
assh() { 
  local host="${1}"
  local studio="${2}"
  code --remote ssh-remote+"$host" /teamspace/studios/this_studio/"$studio"/
}
```

1.5 Start the container
```bash
⚡~ cd hand-rl
⚡~/hand-rl ./job.sh sim-isaac dev
```
The above command starts the container `sim-isaac` in `dev` mode, opens a tty into the container and mounts the project source code.

1.6 Attach VS Code to the container.
`Ctrl+Shift+p` in VS Code and type in `Dev Container: Attach to Running Container`. Make sure extensions Dev Containers are already setup.

1.7 `hand-rl` repo should in `/workspace` and so is `isaaclab` folder.

1.8 Run an example provided by IsaacLab, from inside the container
```bash
$~ cd /workspace/isaaclab
$~/isaaclab isaaclab -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64 --headless --video
``` 
You can see the output in the `/log` directory
1.9 Push scene
```bash
$~ cd /workspace/hand-rl
$~/hand-rl isaaclab -p scripts/vanilla/train.py
```
Configs in `scripts/sample.yaml` is loaded during runtime