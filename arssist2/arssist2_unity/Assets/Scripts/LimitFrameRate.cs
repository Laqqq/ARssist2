using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LimitFrameRate : MonoBehaviour
{
    public int TargetFrameRate = 60;
    private void Start()
    {
        QualitySettings.vSyncCount = 2;
        Application.targetFrameRate = TargetFrameRate;
    }
}
