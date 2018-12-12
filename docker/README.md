# Building the Docker Image

To run the dotasevice in a docker container, it makes sense to have two distinct images:

- Dota image with just the game.
- Image inheriting from previous Dota game, added with the DotaService.

## Dota Container

Set `STEAM_ID` and `STEAM_PWD` in your env, and run:

```sh
docker build -t dota . -f docker/Dockerfile-dota --build-arg user=$STEAM_ID --build-arg pwd=$STEAM_PWD --build-arg guard=
>>> (...)
>>> Logging in user '$STEAM_ID' to Steam Public...Login Failure: Account Logon Denied 
```

That means you need your guard code. Check your authenticator (email/phone) for the code, e.g.
`ABC123`, now run again (quickly, before the code expires!)
```sh
docker build -t dota . -f docker/Dockerfile-dota --build-arg user=$STEAM_ID --build-arg pwd=$STEAM_PWD --build-arg guard=ABC123 
```
This will now install ~20GB Dota in all its glory, although for a dedicated server you only need a few
hundred megs. Make sure you have at least ~40GB available.

## DotaService Container

```sh
docker build -t dotaservice . -f docker/Dockerfile-dotaservice
```

# Run

Running the service is trivial. Arg `-d` is for detached, `-p` exposes ports.
Note everything you append to the command is forwarded as arguments to the dotaservice.

To run two dockerservice instances, one on port `13337` and one on `13338`, f.e. run:

```sh
docker run -dp 13337:13337 dotaservice
docker run -dp 13338:13337 dotaservice
```

Or for development run with `it` instead of detached `d` as:
```
docker run -itp 13337:13337 dotaservice
```

You can now open as many dota clients as you want, provided they expose on different ip:ports
combinations and they don't collide.

Creating an MacOS-native mount seems very detrimental to speed; CPU (8 containers) goes from 50% to 80%,
it seems fine if the container just puts its actions in `/tmp`.
