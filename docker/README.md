# Building the Docker Image

Set `STEAM_ID` and `STEAM_PWD` in your env, and run:

```sh
docker build -t ds . -f docker/Dockerfile --build-arg user=$STEAM_ID --build-arg pwd=$STEAM_PWD --build-arg guard=
>>> (...)
>>> Logging in user '$STEAM_ID' to Steam Public...Login Failure: Account Logon Denied 
```

That means you need your guard code. Check your authenticator (email/phone) for the code, e.g.
`ABC123`, now run again (quickly, before the code expires!)
```sh
docker build -t ds . -f docker/Dockerfile --build-arg user=$STEAM_ID --build-arg pwd=$STEAM_PWD --build-arg guard=ABC123 
```
This will now install ~20GB Dota in all its glory, although for a dedicated server you only need a few
hundred megs. Make sure you have at least ~40GB available.

Running the service is trivial. Arg `-d` is for detached, `-p` exposes ports and `--port` is one
of the arguments forwarded into the dotaservice (`ds`)

To run two dockerservice instances, one on port `13337` and one on `13338`, f.e. run:

```sh
docker run -dp 13337:13337 ds
docker run -dp 13338:13337 ds
```

You can now open as many dota clients as you want, provided they expose on different ip:ports
combinations and they don't collide.
