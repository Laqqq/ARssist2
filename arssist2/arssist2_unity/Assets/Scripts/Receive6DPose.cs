using System;
using System.Collections.Generic;
using UnityEngine;
using DVRK;


// Data class for receiving JSON
[Serializable]
public class Pose6D
{
    public Position3D pos;
    public Rotation3D rot;
}

[Serializable]
public class Position3D
{
    public float x;
    public float y;
    public float z;
}

[Serializable]
public class Rotation3D
{
    public float x;
    public float y;
    public float z;
    public float w;
}


public class Receive6DPose : MonoBehaviour
{
    public UDPClient udpclient;

    [Tooltip("blah blah blah")]
    public GameObject root = null;
    public bool useThresholdLerp = true;

    public static float lerp = 0.06f;
    public static float positionJumpThreshold = 0.08f;
    public static float rotationJumpThreshold = 24f;
    public static float positionRecoverThreshold = 0.04f;
    public static float rotationRecoverThreshold = 12f;

    private List<Vector3> pendingPositionList = new List<Vector3>();
    private List<Quaternion> pendingRotationList = new List<Quaternion>();

    public void SetPoseWithThresholdLerp(Matrix4x4 targetMatrix)
    {
        // Linear interpolates between new poses if they are close together. If the pose difference
        // is greather than the thresholds, interpolation is skipped

        Vector3 previousPosition = root.transform.localPosition;
        Quaternion previousRotation = root.transform.localRotation;
        //Vector3 previousScale = transform.localScale;
        Vector3 targetPosition = ARUWPUtils.PositionFromMatrix(targetMatrix);
        Quaternion targetRotation = ARUWPUtils.QuaternionFromMatrix(targetMatrix);
        //Vector3 targetScale = ARUWPUtils.ScaleFromMatrix(targetMatrix);

        float positionDiff = Vector3.Distance(targetPosition, previousPosition);
        float rotationDiff = Quaternion.Angle(targetRotation, previousRotation);

        if (Mathf.Abs(positionDiff) < positionJumpThreshold && Mathf.Abs(rotationDiff) < rotationJumpThreshold)
        {
            root.transform.localRotation = Quaternion.Slerp(previousRotation, targetRotation, lerp);
            root.transform.localPosition = Vector3.Lerp(previousPosition, targetPosition, lerp);
            //transform.localScale = Vector3.Lerp(previousScale, targetScale, lerp);
            pendingPositionList.Clear();
            pendingRotationList.Clear();
        }
        else
        {
            // maybe there is a jump
            pendingPositionList.Add(targetPosition);
            pendingRotationList.Add(targetRotation);
            bool confirmJump = true;
            if (pendingPositionList.Count > 15)
            {
                for (int i = 0; i < 14; i++)
                {
                    float tempPositionDiff = Vector3.Distance(pendingPositionList[pendingPositionList.Count - i - 1], pendingPositionList[pendingPositionList.Count - i - 2]);
                    float tempRotationDiff = Quaternion.Angle(pendingRotationList[pendingRotationList.Count - i - 1], pendingRotationList[pendingRotationList.Count - i - 2]);
                    if (Mathf.Abs(tempPositionDiff) > positionRecoverThreshold || Mathf.Abs(tempRotationDiff) > rotationRecoverThreshold)
                    {
                        confirmJump = false;
                        break;
                    }
                }
                if (confirmJump)
                {
                    root.transform.localRotation = targetRotation;
                    root.transform.localPosition = targetPosition;
                    pendingPositionList.Clear();
                    pendingRotationList.Clear();
                }
            }
        }

    }



    // Update is called once per frame
    void Update()
    {
        string msg = udpclient.GetLatestUDPPacket();
        if (msg != "")
        {
            Pose6D p = JsonUtility.FromJson<Pose6D>(msg);

            Vector3 position = new Vector3(p.pos.x, p.pos.y, p.pos.z);
            Quaternion rotation = new Quaternion(p.rot.x, p.rot.y, p.rot.z, p.rot.w);






            // If root is not null, then the root object pose will be set so that this object, the 'child' will
            // be in the desired position.
            if (root != null)
            {
                // this object is the child.
                Matrix4x4 desiredChildPose = Matrix4x4.TRS(position, rotation, this.transform.lossyScale);

                // We need to set the root of the entire robot tree so that the marker is in the place.
                // Because the robot kinematics may change the transform between the RCM and the Marker
                // we need to calculate the pose of the parent
                Matrix4x4 childRootInverse = this.transform.worldToLocalMatrix;
                Matrix4x4 parentRoot = root.transform.localToWorldMatrix;
                Matrix4x4 parentToChildInverse = childRootInverse * parentRoot;

                Matrix4x4 targetMatrix = desiredChildPose * parentToChildInverse;

                if (useThresholdLerp)
                {
                    SetPoseWithThresholdLerp(targetMatrix);
                }
                else
                {
                    // Debugging. Instead of setting the pose directly, use aruwpTargetRoot.SetNextPose above because
                    // it does lerping.
                    Vector3 parentPosition = ARUWPUtils.PositionFromMatrix(targetMatrix);
                    Quaternion parentRotation = ARUWPUtils.QuaternionFromMatrix(targetMatrix);
                    root.transform.position = parentPosition;
                    root.transform.rotation = parentRotation;
                }

            }
            else
            {
                // If root is null, then just set the pose of this object directly (no parent stuff)
                this.transform.position = position;
                this.transform.rotation = rotation;
            }

        }
    }
}
