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
using UnityEngine;
using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;

using System.Net;
using System.Net.Sockets;
using System.Threading;





public class UDPServer : MonoBehaviour {

    public int port = 41234;
    public string dest = "127.0.0.1";

    private UdpClient udpClient;

    public void Start()
    {
        udpClient = new UdpClient();
    }

    public void FixedUpdate()
    {
        string pose = this.name + "\n" + transform.worldToLocalMatrix.ToString() + "\n";
        //Debug.Log(pose);
        SendUDPPacket(pose);
    }


    public bool SendUDPPacket(string message) {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);

            udpClient.Send(data, data.Length, dest, port);
        }
        catch (Exception e)
        {
            Debug.LogError(e);
        }
        return true;   
    }

    public void OnDestroy()
    {
        udpClient.Close();
    }
}
