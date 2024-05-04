/*
    James William Fletcher (github.com/mrbid)
        April 2024
*/

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <sys/file.h>
#include <unistd.h>

#include "rply.h"
#include "../icosphere.h"

#define uint unsigned int
#define INPUT_SIZE 2
#define OUTPUT_SIZE 6

#pragma GCC diagnostic ignored "-Wunused-result"

//*************************************
// utility functions
//*************************************
void timestamp(char* ts){const time_t tt = time(0); strftime(ts, 16, "%H:%M:%S", localtime(&tt));}
float urandf32()
{
    static const float RECIP_FLOAT_UINT32_MAX = 1.f/(float)((unsigned int)-1);
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    unsigned int s = 0;
    read(f, &s, sizeof(unsigned int));
    close(f);
    return ((float)s) * RECIP_FLOAT_UINT32_MAX;
}
void writeWarning(const char* s)
{
    FILE* f = fopen("WARNING_FLAGGED_ERROR.TXT", "a"); // just make it long so that it is noticable
    if(f != NULL)
    {
        char strts[16];
        timestamp(&strts[0]);
        fprintf(f, "[%s] %s\n", strts, s);
        printf("[%s] %s\n", strts, s);
        fclose(f);
    }
}

//*************************************
// process models
//*************************************
// permanent 8 MB staging buffer
#define MAX_SIZE 2097152
float vbuff[MAX_SIZE];
uint vbl = 0; // vertex buffer length
uint nverts = 0; // number of vertices
static int vertex_cb(p_ply_argument argument)
{
    if(vbl > MAX_SIZE-1)
    {
        puts("terminated: model too big for internal staging vertex buffer");
        return 0;
    }
    static uint vc = 0;
    long eol;
    ply_get_argument_user_data(argument, NULL, &eol);
    vbuff[vbl] = ply_get_argument_value(argument);
    if(vc > 5){vbuff[vbl] *= 0.003921569f;}
    vbl++;
    vc++;
    if(eol){vc = 0;}
    return 1;
}

//*************************************
// process entry point
//*************************************
int main(int argc, char** argv)
{
    if(argc != 2){return 0;}

    char load_file[512];
    sprintf(load_file, "../ply/%s", argv[1]);
    //writeWarning(load_file);

    char input_file[512];
    sprintf(input_file, "xp/%s", argv[1]);
    //writeWarning(input_file);

    char output_file[512];
    sprintf(output_file, "yp/%s", argv[1]);
    //writeWarning(output_file);

    const size_t INPUT_SIZE_BYTES = INPUT_SIZE*sizeof(float);
    const size_t OUTPUT_SIZE_BYTES = OUTPUT_SIZE*sizeof(float);
    float input[INPUT_SIZE];
    float output[OUTPUT_SIZE];

    // reset buffers
    vbl = 0;
    const float model_seed = urandf32();

    // open file
    p_ply ply = ply_open(load_file, NULL, 0, NULL);
    if(!ply){writeWarning("open fail"); return 0;} // rply spits an error to console anyway if the open fails
    if(!ply_read_header(ply))
    {
        ply_close(ply);
        writeWarning("read header fail");
        return 0;
    }

    // read file setup
    nverts = ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "nx", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "ny", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "nz", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "red", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "green", vertex_cb, NULL, 0);
    ply_set_read_cb(ply, "vertex", "blue", vertex_cb, NULL, 1);

    // read file
    if(!ply_read(ply))
    {
        ply_close(ply);
        writeWarning("read fail");
        return 0;
    }

    // close file
    ply_close(ply);

    // process vertices (36 bytes / 9 floats per vertex)
    for(uint i = 0; i < nverts; i++)
    {
        // printf("x: %f\n",  vbuff[9*i]);
        // printf("y: %f\n",  vbuff[(9*i)+1]);
        // printf("z: %f\n",  vbuff[(9*i)+2]);
        // printf("nx: %f\n", vbuff[(9*i)+3]);
        // printf("ny: %f\n", vbuff[(9*i)+4]);
        // printf("nz: %f\n", vbuff[(9*i)+5]);
        // printf("r: %f\n",  vbuff[(9*i)+6]);
        // printf("g: %f\n",  vbuff[(9*i)+7]);
        // printf("b: %f\n",  vbuff[(9*i)+8]);
        // printf("---\n");
        memcpy(&output[0], &vbuff[9*i], 12);
        memcpy(&output[3], &vbuff[(9*i)+6], 12);
        // printf("x: %f\n", ((float*)output)[0]);
        // printf("y: %f\n", ((float*)output)[1]);
        // printf("z: %f\n", ((float*)output)[2]);
        // printf("r: %f\n", ((float*)output)[3]);
        // printf("g: %f\n", ((float*)output)[4]);
        // printf("b: %f\n", ((float*)output)[5]);
        // printf("###\n");

        // get vertex pos
        const float vx = vbuff[9*i];
        const float vy = vbuff[(9*i)+1];
        const float vz = vbuff[(9*i)+2];
        //printf("v: %f %f %f\n", vx, vy, vz);

        // get vertex normal
        const float vnx = vbuff[(9*i)+3];
        const float vny = vbuff[(9*i)+4];
        const float vnz = vbuff[(9*i)+5];
        //printf("vn: %f %f %f\n", vnx, vny, vnz);

        // get remainder distance of vertex from 0,0,0
        const float pd = 1.f-sqrtf(vx*vx + vy*vy + vz*vz);
        //printf("pd: %f\n", pd);

        // scale the vertex normal by remainder
        const float pdx = vnx*pd;
        const float pdy = vny*pd;
        const float pdz = vnz*pd;
        //printf("vn-s: %f %f %f\n", pdx, pdy, pdz);

        // project vertex positon to unit sphere
        const float upx = vx+pdx;
        const float upy = vy+pdy;
        const float upz = vz+pdz;
        //printf("v-p: %f %f %f\n", upx, upy, upz);

        // find closest vertex on the ico sphere
        int closest = -1;
        float lcd = 9999.f;
        for(uint i = 0; i < icosphere_size; i+=3)
        {
            const float xm = icosphere[i]   - upx;
            const float ym = icosphere[i+1] - upy;
            const float zm = icosphere[i+2] - upz;
            const float d = sqrtf(xm*xm + ym*ym + zm*zm);
            if(d < lcd){lcd = d; closest=i;}
        }
        // printf("a: %f %f %f\n", icosphere[closest], icosphere[closest+1], icosphere[closest+2]);
        // printf("b: %f %f %f\n", upx, upy, upz);
        // printf("lcd: %f\n", lcd);
        if(closest == -1)
        {
            puts("found no closest unit sphere point on icosphere");
            continue;
        }

        input[0] = model_seed; // random
        input[1] = ((float)closest) / ((float)icosphere_size); // icosphere index
        //if(input[1] > 1.f){printf("ind: %f\n", input[1]);}
        // input[2] = -vnx; // inverted vertex normal
        // input[3] = -vny;
        // input[4] = -vnz;

        // write processed vertex
        FILE* fp = fopen(input_file, "ab");
        if(fp != NULL)
        {
            if(fwrite(&input, 1, INPUT_SIZE_BYTES, fp) != INPUT_SIZE_BYTES)
                puts("train_x.dat: write error.");
            fclose(fp);
        }
        fp = fopen(output_file, "ab");
        if(fp != NULL)
        {
            if(fwrite(&output, 1, OUTPUT_SIZE_BYTES, fp) != OUTPUT_SIZE_BYTES)
                puts("train_y.dat: write error.");
            fclose(fp);
        }
    }

    return 0;
}
