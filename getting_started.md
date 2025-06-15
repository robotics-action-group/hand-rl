
1.1 Login to Lightning Studio ([lightning.ai/signup](https://lightning.ai/sign-up), use university email for quick approval) and start a studio with a GPU. Make sure to connect your github account.

## Remeber to do the git integration, and restart terminal when you do.

1.2 Connect to that studio from terminal.

## The key can be found on the side with the terminal icon.

In your computer:
```bash
$ ssh <key>@ssh.lightning.com
```

1.3 Clone this repository inside studio and open in VS Code.

Lighting Terminal:
```bash
⚡~ git clone https://github.com/robotics-action-group/hand-rl.git
```

In your computer:
```bash
code --folder-uri=vscode-remote://ssh-remote+<key>@ssh.lightning.ai/teamspace/studios/this_studio/hand-rl
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
Lighting Terminal:
```bash
⚡~ cd hand-rl
⚡~/hand-rl ./job.sh sim-isaac dev
```
`dev` mode starts the container `sim-isaac`, open a tty into the container and mount the project source code.

1.6 Attach VS Code to the container.
`Ctrl+Shift+p` in VS Code and type in `Dev Container: Attach to Running Container`, make sure extensions for Dev Container are setup.

1.7 `hand-rl` repo should in `/workspace` and so is `isaaclab` folder.

1.8 Run an example provided by IsaacLab, from inside the container
Lighting Terminal:
```bash
$~ cd /workspace/isaaclab
$~/isaaclab isaaclab -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64 --headless --video
``` 

1.9 Push scene
Lighting Terminal:
```bash
$~ cd /workspace/hand-rl
$~/hand-rl isaaclab -p scripts/vanilla/train.py
```
Configs in `scripts/sample.yaml` is loaded during runtime
