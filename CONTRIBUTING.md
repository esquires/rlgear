# How to contribute

## Bug reports or feature requests

[Open an issue](https://github.com/esquires/rlgear/issues) with your bug report
or feature request.

## Contributing code

1. [Create a fork](https://github.com/esquires/rlgear/fork)

2. Before creating a new commit, make sure your changes pass the tests. All
   pull requests are run through travis prior to being merged so please make
   sure all tests pass in the docker command below:

```
# run individual tests
python tests/test_style.py
python tests/test_utils.py
python tests/test_train_cartpole.py

# run all tests
py.test-3 tests

# run in docker container (this is what is run when a pull request is made
# so it is recommended to run this locally prior to making a pull request)
docker build -t rlgear:latest .
```

3. Push a commit with a meaningful message to your own fork that is based off
   of the latest master branch

4. If you have multiple commits related to the same feature, please consider
   squashing your commits into a single commit (e.g., git rebase -i HEAD~10)

5. [Submit a pull request](https://github.com/gtri/rlgear/compare)
