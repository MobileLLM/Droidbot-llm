# DroidBot-LLM

## About

(This is an ongoing work. Repo not ready yet.)

DroidBot-LLM is an LLM-powered test input generator for Android.
DroidBot-LLM is based on [DroidBot](https://github.com/honey/droidbot). By leveraging LLM, the generated input can be more intelligent.

**Reference**

[Li, Yuanchun, et al. "DroidBot: a lightweight UI-guided test input generator for Android." In Proceedings of the 39th International Conference on Software Engineering Companion (ICSE-C '17). Buenos Aires, Argentina, 2017.](http://dl.acm.org/citation.cfm?id=3098352)

## Prerequisite

1. `Python` (both 2 and 3 are supported)
2. `Java`
3. `Android SDK`
4. Add `platform_tools` directory in Android SDK to `PATH`
5. Set `GPT_API_URL` and `GPT_API_KEY` environment variables to your ChatGPT API url and key.

## How to install

Clone this repo, change directory into the root directory, and install with `pip install -e .`. If successfully installed, you should be able to execute `droidbot -h`.

Alternatively, you can run without installation using `python start.py` in the root directory.

## How to use

1. Make sure you have:

    + `.apk` file path of the app you want to analyze.
    + A device or an emulator connected to your host machine via `adb`.

2. Start DroidBot with LLM-guided policy:

    ```
    droidbot -a <path_to_apk> -o output_dir -policy llm_guided
    ```
    That's it! You will find much useful information, including the UTG, generated in the output dir.

    + If you are using multiple devices, you may need to use `-d <device_serial>` to specify the target device. The easiest way to determine a device's serial number is calling `adb devices`.
    + On some devices, you may need to manually turn on accessibility service for DroidBot (required by DroidBot to get current view hierarchy).
    + If you want to test a large scale of apps, you may want to add `-keep_env` option to avoid re-installing the test environment every time.
    + You may find other useful features in `droidbot -h`.

3. Start DroidBot with manual policy:
    go to `start.py` and set `os.environ['manual'] = 'True'`
    ```
    droidbot -a <path_to_apk> -o output_dir -policy llm_guided
    ```
    That's it! You will find much useful information, including the UTG, generated in the output dir.

## Acknowledgement

1. The DroidBot Project: [DroidBot](https://github.com/honey/droidbot)

