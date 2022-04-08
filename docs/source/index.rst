.. OpenBackdoor documentation master file, created by
   sphinx-quickstart on Thu Mar 31 16:08:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenBackdoor's documentation!
===================================

**OpenBackdoor** is an open-scource toolkit for textual backdoor attack and defense, which enables easy implementation, evaluation, and extension of both attack and defense models.

OpenBackdoor has the following features:

- **Extensive implementation** OpenBackdoor implements 11 attack methods along with 4 defense methods, which belong to diverse categories. Users can easily replicate these models in a few line of codes. 
- **Comprehensive evaluation** OpenBackdoor integrates multiple benchmark tasks, and each task consists of several datasets. Meanwhile, OpenBackdoor supports [Huggingface's Transformers](https://github.com/huggingface/transformers) and [Datasets](https://github.com/huggingface/datasets) libraries.

- **Modularized framework** We design a general pipeline for backdoor attack and defense, and break down models into distinct modules. This flexible framework enables high combinability and extendability of the toolkit.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   notes/installation.md
   notes/usage.md
   notes/config.md


.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   modules/attacker
   modules/defender
   modules/dataset
   modules/poisoner
   modules/trainer
