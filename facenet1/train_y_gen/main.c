/*
    James William Fletcher (github.com/mrbid)
        April 2024
*/

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "rply.h"

#define uint unsigned int
#define OUTPUT_SIZE 32768
#define toindex(x,y,z) (z * 1024) + (y * 32) + x
#pragma GCC diagnostic ignored "-Wunused-result"
void timestamp(char* ts){const time_t tt = time(0); strftime(ts, 16, "%H:%M:%S", localtime(&tt));}

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
    vbl++;
    return 1;
}
void processModels(const char* dir_path)
{
    const size_t OUTPUT_SIZE_BYTES = OUTPUT_SIZE*4;
    float output[OUTPUT_SIZE];
    float output_acc[OUTPUT_SIZE];
    unsigned char output_uint8[OUTPUT_SIZE];
    for(uint h=1; h <= 3333; h++)
    {
        // reset buffers
        vbl = 0;

        // new file load path
        char fp[384];
        sprintf(fp, "%s/%i.ply", dir_path, h);
        //puts(fp);

        // open file
        p_ply ply = ply_open(fp, NULL, 0, NULL);
        if(!ply){continue;} // rply spits an error to console anyway if the open fails
        if(!ply_read_header(ply))
        {
            ply_close(ply);
            continue;
        }

        // read file setup
        nverts = ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
        ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 0);
        ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 0);
        ply_set_read_cb(ply, "vertex", "red", vertex_cb, NULL, 0);
        ply_set_read_cb(ply, "vertex", "green", vertex_cb, NULL, 0);
        ply_set_read_cb(ply, "vertex", "blue", vertex_cb, NULL, 1);

        // read file
        if(!ply_read(ply))
        {
            ply_close(ply);
            continue;
        }

        // close file
        ply_close(ply);

        // find longest axis
        float lax = 0.f;
        for(uint i = 0; i < nverts; i++)
        {
            const float vx = fabsf(vbuff[6*i]);
            const float vy = fabsf(vbuff[(6*i)+1]);
            const float vz = fabsf(vbuff[(6*i)+2]);
            if(vx > lax){lax = vx;}
            if(vy > lax){lax = vy;}
            if(vz > lax){lax = vz;}
        }
        const float rlax = 1.f/lax;
        //printf("lax: %f\n", lax);

        // process vertices into a grayscale volume
        for(uint i = 0; i < nverts; i++)
        {
            // printf("x: %f\n",  vbuff[6*i]);
            // printf("y: %f\n",  vbuff[(6*i)+1]);
            // printf("z: %f\n",  vbuff[(6*i)+2]);
            // printf("r: %f\n",  vbuff[(6*i)+3]);
            // printf("g: %f\n",  vbuff[(6*i)+4]);
            // printf("b: %f\n",  vbuff[(6*i)+5]);
            // printf("###\n");

            // get scaled vertex pos
            const float vx = (((vbuff[6*i]     * rlax) + 1.f) * 0.5f) * 31.f;
            const float vy = (((vbuff[(6*i)+1] * rlax) + 1.f) * 0.5f) * 31.f;
            const float vz = (((vbuff[(6*i)+2] * rlax) + 1.f) * 0.5f) * 31.f;
            //printf("v: %f %f %f\n", vx, vy, vz);

            // round it to volume
            const uint vox = (uint)roundf(vx);
            const uint voy = (uint)roundf(vy);
            const uint voz = (uint)roundf(vz);
            //printf("vo: %i %i %i\n", vox, voy, voz);
            // if(vox > 31 || voy > 31 || voz > 31)
            // {
            //     puts("PROBLEM");
            //     return;
            // }

            // get vertex color
            const float cx = vbuff[(6*i)+3];
            const float cy = vbuff[(6*i)+4];
            const float cz = vbuff[(6*i)+5];
            //printf("c: %f %f %f\n", cx, cy, cz);

            // accumulate normalised grayscale color
            output[toindex(vox, voy, voz)] += (cx + cy + cz) / 765.f;
            output_acc[toindex(vox, voy, voz)] += 1.f;
            //printf("%f\n", (cx + cy + cz) / 765.f);
        }

        // sum volume colors
        for(uint i = 0; i < OUTPUT_SIZE; i++)
        {
            output[i] /= output_acc[i];
            output_uint8[i] = (unsigned char)roundf(output[i]*255.f);
        }

        // append volume
        FILE* f = fopen("train_y.dat", "ab");
        if(f != NULL)
        {
            if(fwrite(&output_uint8, 1, OUTPUT_SIZE, f) != OUTPUT_SIZE)
                puts("train_y.dat: write error.");
            fclose(f);
        }

        // stats
        char strts[16];
        timestamp(&strts[0]);
        printf("[%s] [%u] %s\n", strts, h, fp);
    }
}

//*************************************
// process entry point
//*************************************
int main(int argc, char** argv)
{
    const time_t st = time(0);
    processModels("../ply");
    printf("Time Taken: %lu seconds\n", time(0)-st);
    return 0;
}
