## Streaming Joint Positions to HoloLens 2
The joint values are sent from the dVRK to the HoloLens 2 via UDP using the code in this repository: https://github.com/jhu-saw/sawSocketStreamer

Note: I only personally tested sawSocketStreamer on ROS 1, but it should work on ROS 2 as well  

### sawSocketStreamer installation (ROS 1)
Start by building a ROS 1 workspace of the dVRK software. Installation documentation can be found here: https://dvrk.readthedocs.io/2.3.0/pages/software/compilation/ros1.html

After building successfully, clone the [sawSocketStreamer](https://github.com/jhu-saw/sawSocketStreamer) repository into `catkin_ws/src/cisst-saw` and run `catkin build` again.

for ROS 2, consult the [documentation](https://dvrk.readthedocs.io/2.3.0/pages/software/compilation/ros2.html) for building

### sawSocketStreamer configuration
Full configuration file details are available in the [sawSocketStreamer](https://github.com/jhu-saw/sawSocketStreamer) repository.

The imporant thing is to set the IP address of the HoloLens 2

In the `socketStreamerConfig` directory, you will find four configuration files:
- `streamerECM-ARssist.json`
- `streamerPSM1-ARssist.json`
- `streamerPSM2-ARssist.json`
- `manager-socket-streamer-ARssist.json`

in the three files beginning with `streamer` you must edit the IP address of the Hololens.

The four files must be in the same directory

### Running sawSocketStreamer
As normal, you must run `sawIntuitiveResearchKitQtConsoleJSON` with the appropriate JSON configuration for your local dVRK setup and for the PSM1, PSM2, and ECM (and SUJ). For example, at JHU we would run: 

`sawIntuitiveResearchKitQtConsoleJSON -j catkin_ws/src/dvrk/dvrk_config_jhu/jhu-daVinci/console-SUJ-ECM-PSM1-PSM2.json`

now, to include the sawSocketStreamer component, we can add the path to the manager JSON file with the flag `-m`.  For example:

 `-m manager-socket-streamer-ARssist.json`

so the full command is:

`sawIntuitiveResearchKitQtConsoleJSON -j catkin_ws/src/dvrk/dvrk_config_jhu/jhu-daVinci/console-SUJ-ECM-PSM1-PSM2.json -m manager-socket-streamer-ARssist.json`

## Data Format
Over the UDP connection, different JSON messages will be sent. We only care about the messages containing `"measured_js"` and `"jaw/measured_js"` which are shown below:

```
{
  "measured_js": {
    "AutomaticTimestamp": true,
    "Effort": [
      -0.004980754239783524,
      0.0035392203202758753,
      -0.40980161585186925,
      -0.0000690404422177524,
      0.000830556841949222,
      -0.00030587982854088044
    ],
    "Name": [
      "yaw",
      "pitch",
      "insertion",
      "roll",
      "wrist_pitch",
      "wrist_yaw"
    ],
    "Position": [
      -0.472492242710804,
      -0.23289445528798855,
      0.09510101956000001,
      -0.00482286870014406,
      -0.00314263949460513,
      -0.016795249628295556
    ],
    "Timestamp": 46.072619962,
    "Valid": true,
    "Velocity": [
      0,
      0,
      0,
      0,
      0,
      0
    ]
  }
}
```
```
{
  "jaw/measured_js": {
    "AutomaticTimestamp": false,
    "Effort": [
      -0.00021319451932491846
    ],
    "Name": [
      "jaw"
    ],
    "Position": [
      0.024664953986276168
    ],
    "Timestamp": 46.082467158,
    "Valid": true,
    "Velocity": [
      0
    ]
  }
}
```