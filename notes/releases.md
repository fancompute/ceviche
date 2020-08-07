# Release

Instructions / notes for tagging a new release for ceviche.

## What's the point of this?

As we add commits to `ceviche`, they update the current version on github.

If someone uses `git clone` to clone the most recent version, they will see these changes.

But if someone uses `pip install ceviche`, they will get the last version that was uploaded to the PyPI website where `pip` gets the code to download.

So when we find a good place to save our progress, we will 'release' a new version of ceviche and upload this version to PyPI so it can be downloaded with `pip`.

These are the steps to do this, for future reference.

## Instructions

You need to do the following things.

### Github Changes

from the main github repository page for `ceviche` click

    releases -> draft new release

then add the fields for this release, for example, for version `x.y.z`.

set `tag version` to `x.y.z`.

give it a descriptive name.

click `publish release`.

### File Changes

Now we need to change the actual code to point to release version `x.y.z`.

Change `ceviche/setup.py` to have the new release version:

    version='x.y.z'

Change `ceviche/ceviche/__init__.py` to have the new release version:

    __version__ = 'x.y.z'

Commit changes and push them to master.

    git add .
    git commit -m"release x.y.z"
    git push origin master

### Uploading to PyPI

You'll need a PyPI account and have maintainer access to ceviche, which is located [here](https://pypi.org/project/ceviche/).

Let me know if you need this and I'll give it to you.

first, you'll need an empty `ceviche/dist` directory.

    rm -rf dist/

if needed, and then

    mkdir dist

Now do

    python setup.py sdist bdist_wheel
    twine upload dist/*

I'm assuming you're using python3, if you're not, you should be.

You might need to install twine for this to work.
There are a few ways to do it, this works for me.

This should prompt you to sign into PyPI through your terminal, after which it will upload to the website.

