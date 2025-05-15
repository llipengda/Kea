<div align="center">
<h1>LAMP: Large-language-model Augmented Mobile Testing Path Exploration Based on Kea</h1>

 <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-orange'></a> &nbsp;&nbsp;&nbsp;
 <a><img src='https://img.shields.io/badge/python-3.8, 3.9, 3.10, 3.11, 3.12, 3.13-blue'></a> &nbsp;&nbsp;&nbsp;
</div>

<div align="center">
    <img src="kea/resources/Lamp.jpg" alt="kea_logo" style="border-radius: 18px"/>
</div>


### Introduction

Lamp is a newly-developed path exploration strategy for Kea, which is a general and practical testing tool based on [property-based testing](https://en.wikipedia.org/wiki/Software_testing#Property_testing) for finding functional (logic) bugs in mobile (GUI) apps.

<p align="center">
  <img src="kea/resources/kea-platforms.jpg" width="300"/>
</p>

### Kea 

📘 **[Kea won ACM Distinguished Paper Award in ASE 2024)](https://xyiheng.github.io//files/Property_Based_Testing_for_Android_Apps.pdf)**

Originally, Kea provides three path exploration strategies:

Random Strategy – explores pages randomly.
Guided Strategy – explores pages follows main path.
LLM Strategy – uses a language model to generate inputs for reaching hard-to-access UI states.

We have added a new strategy called LAMP, which is based on the LLM strategy but with a more efficient and effective path exploration strategy.

## :rocket: Quick Start

**Prerequisites**

- Python 3.8+
- `adb` or `hdc` cmd tools available
- Connect an Android / HarmonyOS device or emulator to your PC

[The setup guide for Android / HarmonyOS envirnments.](https://kea-technic-docs.readthedocs.io/en/latest/part-keaUserManuel/envirnment_setup.html) (Provided by Kea)

**Installation**

Enter the following commands to install kea.

```bash
git clone https://github.com/Mingle-2012/Kea.git
cd Kea
pip install -e .
```

**Quick Start**

```
kea -f example/example_property.py -a example/omninotes.apk -p new
```

## :wrench: Configuration

We provide two implementations of LAMP:

1. **new strategy**: This strategy improves the original LLM strategy by developing a new UI tarpit detection algorithm and an iterative LLM-based path exploration strategy. It is more efficient and effective than the original LLM strategy. To test this strategy, you need to set the `-p new` parameter when running Kea.

2. **enhanced new strategy**: This strategy further improves the new strategy by adding a newly proposed enhanced random exploration strategy. It improves the code coverage and UI coverage by lowering the probability of exploring the same path. To test this strategy, you need to set the `-p enhance` parameter when running Kea.

### Our contribution

- We propose a semantic-based UI tarpit detection algorithm
- We propose an augmented LLM exploration strategy based on iterative event sequence generation
- We propose a frequency-aware random exploration strategy

To minimize changes to the original codebase, we concentrated the majority of our modifications in the file `kea/input_policy.py`, where you can find the implementations of these new strategies. 

### Contributors

The developers of Lamp are:
[Pengda Li](https://github.com/llipengda), 
[Zekai Wu](https://github.com/Mingle-2012),
[Yunbiao Zhang](https://github.com/sirius-1024),

