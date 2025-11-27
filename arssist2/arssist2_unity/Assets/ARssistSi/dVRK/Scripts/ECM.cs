/*
    Author(s):  Long Qian
    Created on: 2019-03-29
    (C) Copyright 2015-2018 Johns Hopkins University (JHU), All Rights Reserved.

    --- begin cisst license - do not edit ---
    This software is provided "as is" under an open source license, with
    no warranty.  The complete license can be found in license.txt and
    http://www.cisst.org/cisst/license.txt.
    --- end cisst license ---
*/

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DVRK {

    [System.Serializable]
    public class ECMState {
        public JointState GetStateJoint;
    }

    [System.Serializable]
    public class JointState {
        public bool AutomaticTimestamp;
        public float[] Effort;
        public string[] Name;
        public float[] Position;
        public float Timestamp;
        public int[] Type;
        public bool Valid;
        public int[] Velocity;
    }

    [System.Serializable]
    public class JointStateMeasured
    {
        public JointState measured_js;
    }

    public static class JsonHelpers
    {
        public static char? GetFirstKeyInitial(string json)
        {
            for (int i = 0; i < json.Length; i++)
            {
                char c = json[i];
                if (c == '{' || c == '"' || char.IsWhiteSpace(c))
                    continue;
                return c; // this is the first letter of the first key
            }
            return null; // malformed input
        }
        public static int GetFirstSlashIndex(string json)
        {
            for (int i = 0; i < json.Length; i++)
            {
                char c = json[i];
                if (c == '/')
                    return i; // location of the first '/'
            }
            return 0; // not found 
        }
    }



    public class ECM : URDFRobot {
        private bool messageFirstParsed = false;

        private bool CheckConsistency(ECMState state) {
            int currentIndex = 0;
            foreach (URDFJoint joint in independentJoints) {
                if (joint.name.StartsWith(state.GetStateJoint.Name[currentIndex])) {
                    currentIndex++;
                    continue;
                }
                else {
                    Debug.Log("ECM error: " + joint.name + " does not start with " + state.GetStateJoint.Name[currentIndex]);
                    return false;
                }
            }
            Debug.Log("ECM consistency check passed");

            return true;
        }

        public override void HandleMessage(string message) {

            char? firstLetter = JsonHelpers.GetFirstKeyInitial(message);

            // The message format for the new sawSocketStreamer
            // {"measured_js": ...
            // Assuming compact json with no whitespace
            if (firstLetter == 'm')
            {
                JointStateMeasured state = JsonUtility.FromJson<JointStateMeasured>(message);
                int currentIndex = 0;

                // Assuming correct order
                if (state.measured_js.Position != null)
                {
                    foreach (URDFJoint joint in independentJoints)
                    {
                        if (joint.jointType == URDFJoint.JointType.Prismatic)
                        {
                            joint.SetJointValue(state.measured_js.Position[currentIndex]);
                        }
                        else
                        {
                            joint.SetJointValue(state.measured_js.Position[currentIndex] / (float)(Math.PI) * 180f);
                        }
                        currentIndex++;
                    }
                }

            }

            // message format for the old output of sawSocketStreamer
            // {"GetStateJaw": ...
            // Assuming compact json with no whitespace
            else if (firstLetter == 'G')
            {
                ECMState state = JsonUtility.FromJson<ECMState>(message);
                if (!messageFirstParsed)
                {
                    if (!CheckConsistency(state))
                    {
                        messageFirstParsed = false;
                        return;
                    }
                    else
                    {
                        messageFirstParsed = true;
                    }
                }

                // Hack for da Vinci Si in Mock OR
                state.GetStateJoint.Position[0] = -state.GetStateJoint.Position[0];
                state.GetStateJoint.Position[1] = -state.GetStateJoint.Position[1];
                int currentIndex = 0;
                // Assuming correct order
                foreach (URDFJoint joint in independentJoints)
                {
                    if (joint.jointType == URDFJoint.JointType.Prismatic)
                    {
                        joint.SetJointValue(state.GetStateJoint.Position[currentIndex]);
                    }
                    else
                    {
                        joint.SetJointValue(state.GetStateJoint.Position[currentIndex] / (float)(Math.PI) * 180f);
                    }
                    currentIndex++;
                }
            }
        }
    }
}
