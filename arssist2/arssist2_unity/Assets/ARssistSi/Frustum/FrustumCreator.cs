using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FrustumCreator : MonoBehaviour {
    
    public GameObject frustumEnd;
    public float width = 0.592f;
    public float height = 0.333f;
    public float depth = 0.13f;

    // Use this for initialization
    void Start () {
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
        Mesh mesh = new Mesh();

        Vector3[] vertices = new Vector3[5];
        vertices[0] = new Vector3(0, 0, 0);
        vertices[1] = new Vector3(-width * depth, height * depth, depth);
        vertices[2] = new Vector3(width * depth, height * depth, depth);
        vertices[3] = new Vector3(width * depth, -height * depth, depth);
        vertices[4] = new Vector3(-width * depth, -height * depth, depth);

        int[] triangles = new int[12];
        triangles[0] = 0;
        triangles[1] = 1;
        triangles[2] = 2;
        triangles[3] = 0;
        triangles[4] = 2;
        triangles[5] = 3;
        triangles[6] = 0;
        triangles[7] = 3;
        triangles[8] = 4;
        triangles[9] = 0;
        triangles[10] = 4;
        triangles[11] = 1;

        Vector2[] uv = new Vector2[5];
        uv[0] = new Vector2(0.1f, 0.1f);
        uv[1] = new Vector2(1, 1);
        uv[2] = new Vector2(1, 1);
        uv[3] = new Vector2(1, 1);
        uv[4] = new Vector2(1, 1);

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uv;

        meshFilter.mesh = mesh;

        StartModelFrustumEnd();
    }


    private void StartModelFrustumEnd() {
        MeshFilter meshFilterEnd = frustumEnd.GetComponent<MeshFilter>();
        MeshRenderer meshRendererEnd = frustumEnd.GetComponent<MeshRenderer>();
        Mesh meshEnd = new Mesh();
        Vector3[] vertices = new Vector3[4];
        vertices[0] = new Vector3(-width * depth, height * depth, depth);
        vertices[1] = new Vector3(width * depth, height * depth, depth);
        vertices[2] = new Vector3(width * depth, -height * depth, depth);
        vertices[3] = new Vector3(-width * depth, -height * depth, depth);

        int[] triangles = new int[6];
        triangles[0] = 0;
        triangles[1] = 1;
        triangles[2] = 2;
        triangles[3] = 0;
        triangles[4] = 2;
        triangles[5] = 3;

        Vector2[] uv = new Vector2[4];
        uv[0] = new Vector2(0, 1);
        uv[1] = new Vector2(1, 1);
        uv[2] = new Vector2(1, 0);
        uv[3] = new Vector2(0, 0);

        meshEnd.vertices = vertices;
        meshEnd.triangles = triangles;
        meshEnd.uv = uv;

        meshFilterEnd.mesh = meshEnd;
    }

    // Update is called once per frame
    void Update () {
		
	}
}
