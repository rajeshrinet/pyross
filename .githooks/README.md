## Git Hooks

Git Hooks are scripts that automatically run during certain events (i.e
when you push, commit etc). Please copy the file pre-push to `.git/hooks` 
to enable automatic unit testing before pushing.

You can do this by running

```
cp .githooks/pre-push .git/hooks/
cd .git/hooks/
chmod +x pre-push
 ```
