using UnityEngine;
using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;

#if !UNITY_EDITOR && UNITY_METRO
using Windows.Networking.Sockets;
using Windows.Networking.Connectivity;
using Windows.Networking;

#else 
using System.Net;
using System.Net.Sockets;
using System.Threading;
#endif


public class UDPKeyboardInput : MonoBehaviour {

    public int port = 8055;

#if !UNITY_EDITOR && UNITY_METRO
    private string lastReceivedUDPPacket = "";
    private readonly Queue<string> receivedUDPPacketQueue = new Queue<string>();

    DatagramSocket socket;

    async void Start() {
        socket = new DatagramSocket();
        socket.MessageReceived += Socket_MessageReceived;
        HostName IP = null;
        try {
            var icp = NetworkInformation.GetInternetConnectionProfile();

            IP = Windows.Networking.Connectivity.NetworkInformation.GetHostNames()
            .SingleOrDefault(
                hn =>
                    hn.IPInformation?.NetworkAdapter != null && hn.IPInformation.NetworkAdapter.NetworkAdapterId
                    == icp.NetworkAdapter.NetworkAdapterId);

            await socket.BindEndpointAsync(IP, port.ToString());
        }
        catch (Exception e) {
            Debug.Log(e.ToString());
            Debug.Log(SocketError.GetStatus(e.HResult).ToString());
            return;
        }
        Debug.Log("DatagramSocket setup done...");
    }



    // Update is called once per frame
    void Update() {
        ;
    }

    public string GetLatestUDPPacket() {
        string returnedLastUDPPacket = "";
        while (receivedUDPPacketQueue.Count > 0) {
            returnedLastUDPPacket = receivedUDPPacketQueue.Dequeue();
        }
        return returnedLastUDPPacket;
    }


    private async void Socket_MessageReceived(Windows.Networking.Sockets.DatagramSocket sender,
        Windows.Networking.Sockets.DatagramSocketMessageReceivedEventArgs args) {
        //Debug.Log("Received message: ");
        //Read the message that was received from the UDP echo client.
        Stream streamIn = args.GetDataStream().AsStreamForRead();
        StreamReader reader = new StreamReader(streamIn);
        string message = await reader.ReadLineAsync();

        //Debug.Log("Message: " + message);

        lastReceivedUDPPacket = message;
        receivedUDPPacketQueue.Enqueue(message);

    }


    private void OnDestroy() {
        if (socket != null) {
            socket.MessageReceived -= Socket_MessageReceived;
            socket.Dispose();
            Debug.Log("Socket disposed");
        }
    }
#else

    Thread receiveThread;
    UdpClient client;

    string lastReceivedUDPPacket = ""; // should use a Lock
    
    public void Start()
    {
        InitReceiveThread();
    }

    private void InitReceiveThread()
    {
        receiveThread = new Thread(
            new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    // receive thread
    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);
                //Debug.Log("UDPReceive: " + text);
                lastReceivedUDPPacket = text;
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }

    public string GetLatestUDPPacket()
    {
        string UDPPacketToReturn = lastReceivedUDPPacket;
        lastReceivedUDPPacket = "";
        return UDPPacketToReturn;
    }

#endif
}
