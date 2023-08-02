## Contributing to the Codebase

#### 0. Look at Open Issues
We currently utilize GitHub Issues as our project management tool for checkmates. Please do the following:
* Look at our [open issues](https://github.com/alteryx/CheckMates/issues)
* Find an unclaimed issue by looking for an empty `Assignees` field.
* If this is your first time contributing, issues labeled ``good first issue`` are a good place to start.
* If your issue is labeled `needs design` or `spike` it is recommended you provide a design document for your feature
  prior to submitting a pull request (PR).
* Connect your PR to your issue by adding the following comment in the PR body: `Fixes #<issue-number>`


#### 1. Clone repo
The code is hosted on GitHub, so you will need to use Git to clone the project and make changes to the codebase. Once you have obtained a copy of the code, you should create a development environment that is separate from your existing Python environment so that you can make and test changes without compromising your own work environment. Additionally, you must make sure that the version of Python you use is at least 3.8.
* clone with `git clone [https://github.com/alteryx/CheckMates.git]`
* install in edit mode with:
    ```bash
    # move into the repo
    cd checkmates
    # installs the repo in edit mode, meaning changes to any files will be picked up in python. also installs all dependencies.
    make installdeps-dev
    ```

<!--- Note that if you're on Mac, there are a few extra steps you'll want to keep track of.
* We've seen some installs get the following warning when importing checkmates: "UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError". [A known workaround](https://stackoverflow.com/a/61531555/841003) is to run `brew reinstall readline xz` before installing the python version you're using via pyenv. If you've already installed a python version in pyenv, consider deleting it and reinstalling. v3.9.7 is known to work. --->

#### 2. Implement your Pull Request

* Implement your pull request. If needed, add new tests or update the documentation.
* Before submitting to GitHub, verify the tests run and the code lints properly
  ```bash
  # runs linting
  make lint

  # will fix some common linting issues automatically, if the above command failed
  make lint-fix

  # runs all the unit tests locally
  make test
  ```

* If you made changes to the documentation, build the documentation to view locally.
  ```bash
  # go to docs and build
  cd docs
  make html

  # view docs locally
  open build/html/index.html
  ```

* Before you commit, a few lint fixing hooks will run. You can also manually run these.
  ```bash
  # run linting hooks only on changed files
  pre-commit run

  # run linting hooks on all files
  pre-commit run --all-files
  ```

Note that if you're building docs locally, the warning suppression code at `docs/source/disable-warnings.py` will not run, meaning you'll see python warnings appear in the docs where applicable. To suppress this, add `warnings.filterwarnings('ignore')` to `docs/source/conf.py`.

#### 3. Submit your Pull Request

* Once your changes are ready to be submitted, make sure to push your changes to GitHub before creating a pull request. Create a pull request, and our continuous integration will run automatically.

* Be sure to include unit tests (and docstring tests, if applicable) for your changes; these tests you write will also be run as part of the continuous integration.

* If your changes alter the following please fix them as well:
    * Docstrings - if your changes render docstrings invalid
    * API changes - if you change the API update `docs/source/api_reference.rst`
    * Documentation - run the documentation notebooks locally to ensure everything is logical and works as intended

* Update the "Future Release" section at the top of the release notes (`docs/source/release_notes.rst`) to include an entry for your pull request. Write your entry in past tense, i.e. "added fizzbuzz impl."

* Please create your pull request initially as [a "Draft" PR](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-pull-requests#draft-pull-requests). This signals the team to ignore it and to allow you to develop. When the checkin tests are passing and you're ready to get your pull request reviewed and merged, please convert it to a normal PR for review.

* We use GitHub Actions to run our PR checkin tests. On creation of the PR and for every change you make to your PR, you'll need a maintainer to click "Approve and run" on your PR. This is a change [GitHub made in April 2021](https://github.blog/2021-04-22-github-actions-update-helping-maintainers-combat-bad-actors/).

* We ask that all contributors sign our contributor license agreement (CLA) the first time they contribute to checkmates. The CLA assistant will place a message on your PR; follow the instructions there to sign the CLA.

Add a description of your PR to the subsection that most closely matches your contribution:
    * Enhancements: new features or additions to CheckMates.
    * Fixes: things like bugfixes or adding more descriptive error messages.
    * Changes: modifications to an existing part of CheckMates.
    * Documentation Changes
    * Testing Changes

If your work includes a [breaking change](https://en.wiktionary.org/wiki/breaking_change), please add a description of what has been affected in the "Breaking Changes" section below the latest release notes. If no "Breaking Changes" section yet exists, please create one as follows. See past release notes for examples of this.
```
.. warning::

    **Breaking Changes**

    * Description of your breaking change
```

## GitHub Issue Guide

* Make the title as short and descriptive as possible.
* Make sure the body is concise and gets to the point quickly.
* Check for duplicates before filing.
* For bugs, a good general outline is: problem summary, reproduction steps, symptoms and scope, root cause if known, proposed solution(s), and next steps.
* If the issue writeup or conversation get too long and hard to follow, consider starting a design document.
* Use the appropriate labels to help your issue get triaged quickly.
* Make your issues as actionable as possible. If they track open discussions, consider prefixing the title with "[Discuss]", or refining the issue further before filing.
