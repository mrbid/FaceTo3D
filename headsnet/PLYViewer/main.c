/*
    James William Fletcher ( github.com/mrbid )
        January 2024

    "If you chase perfection you just end up making something that is so abstract
        and tailored to yourself that no one else will understand it."
*/

#include <dirent.h> 

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h> // getcwd

#define uint GLuint
#define sint GLint

#include "inc/gl.h"
#define GLFW_INCLUDE_NONE
#include "inc/glfw3.h"
#define fTime() (float)glfwGetTime()

#define MAX_MODELS 3752 // hard limit, be aware and increase if needed
#include "inc/esAux5.h"
#include "inc/res.h"
#include "inc/rply.h"

//*************************************
// globals
//*************************************
const char appTitle[]="PLY Viewer";
GLFWwindow* window;
uint winw=1024, winh=768;
float t=0.f,aspect,ww,wh;
mat projection, model;
#define FAR_DISTANCE 66.f
int total_samples = 0;
int current_page = 0;
int total_pages = 0;
float yrot[9] = {0};
const float gs = 1.3f;
const float fd = -4.f;
float selected[MAX_MODELS] = {0};

//*************************************
// utility functions
//*************************************
void timestamp(char* ts){const time_t tt = time(0); strftime(ts, 16, "%H:%M:%S", localtime(&tt));}
void updateModel(){glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (float*)&model.m[0][0]);}
void updateTitle()
{
    char tmp[256];
    sprintf(tmp, "%i / %u", (current_page/9)+1, (total_samples/9)+1);
    glfwSetWindowTitle(window, tmp);
}

//*************************************
// load models from file to gpu memory
//*************************************
// permenant 12 MB staging buffer
#define MAX_SIZE 2097152
GLfloat vertex_buffer[MAX_SIZE];
GLuint index_buffer[MAX_SIZE];
uint vbl = 0, ibl = 0; // buffer lens
uint ntris = 0, nverts = 0;
static int vertex_cb(p_ply_argument argument)
{
    if(vbl > MAX_SIZE-1){return 0;}
    static uint vc = 0;
    long eol;
    ply_get_argument_user_data(argument, NULL, &eol);
    vertex_buffer[vbl] = ply_get_argument_value(argument);
    if(vc > 5){vertex_buffer[vbl] *= 0.003921569f;}
    vbl++;
    vc++;
    if(eol){vc = 0;}
    return 1;
}
static int face_cb(p_ply_argument argument)
{
    if(ibl > MAX_SIZE-1){return 0;}
    long length, value_index;
    ply_get_argument_property(argument, NULL, &length, &value_index);
    switch(value_index)
    {
        case 0:
        case 1: 
            index_buffer[ibl] = ply_get_argument_value(argument);
            ibl++;
            break;
        case 2:
            index_buffer[ibl] = ply_get_argument_value(argument);
            ibl++;
            break;
        default: 
            break;
    }
    return 1;
}
void loadModels(const char* dir_path)
{
    struct dirent *dir;
    DIR* d = opendir(dir_path);
    if(d != NULL)
    {
        while((dir = readdir(d)) != NULL)
        {
            if(dir->d_name[0] != '.')
            {
                // reset buffers
                vbl = 0, ibl = 0;

                // new file load path
                char fp[384];
                sprintf(fp, "%s/%s", dir_path, dir->d_name);

                // open file
                p_ply ply = ply_open(fp, NULL, 0, NULL);
                if(!ply){esModelArray_index++; continue;}
                if(!ply_read_header(ply))
                {
                    ply_close(ply);
                    esModelArray_index++;
                    continue;
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
                ntris = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);

                // read file
                if(!ply_read(ply))
                {
                    ply_close(ply);
                    esModelArray_index++;
                    continue;
                }

                // close file
                ply_close(ply);

                // bind to gpu
                esBind(GL_ARRAY_BUFFER, &esModelArray[esModelArray_index].vid, vertex_buffer, vbl*sizeof(GLfloat), GL_STATIC_DRAW);
                esBind(GL_ELEMENT_ARRAY_BUFFER, &esModelArray[esModelArray_index].iid, index_buffer, ibl*sizeof(GLuint), GL_STATIC_DRAW);
                esModelArray[esModelArray_index].itp = GL_UNSIGNED_INT;
                esModelArray[esModelArray_index].ni = ibl;
                printf("Loaded PLY: %u %u %u\n", esModelArray_index+1, vbl, ibl);
                esModelArray_index++;

                // stats
                total_samples++;
                updateTitle();
                if(total_samples > 3000){break;}
            }
        }
    }
    total_pages = total_samples/9;
    updateTitle();
}

//*************************************
// update & render
//*************************************
void main_loop()
{
    // delta time
    glfwPollEvents();
    t = fTime();

    // clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // render
    mIdent(&model);
    mRotY(&model, d2PI+yrot[0]);
    if(selected[current_page] == 0.f){mRotZ(&model, t*2.f);}else{mRotZ(&model, selected[current_page]);}
    mSetPos(&model, (vec){-gs, gs, fd});
    updateModel();
    esBindRender(current_page);

    mIdent(&model);
    mRotY(&model, d2PI+yrot[1]);
    if(selected[current_page+1] == 0.f){mRotZ(&model, t*2.f);}else{mRotZ(&model, selected[current_page+1]);}
    mSetPos(&model, (vec){0.f, gs, fd});
    updateModel();
    esBindRender(current_page+1);

    mIdent(&model);
    mRotY(&model, d2PI+yrot[2]);
    if(selected[current_page+2] == 0.f){mRotZ(&model, t*2.f);}else{mRotZ(&model, selected[current_page+2]);}
    mSetPos(&model, (vec){gs, gs, fd});
    updateModel();
    esBindRender(current_page+2);

    ///

    mIdent(&model);
    mRotY(&model, d2PI+yrot[3]);
    if(selected[current_page+3] == 0.f){mRotZ(&model, t*2.f);}else{mRotZ(&model, selected[current_page+3]);}
    mSetPos(&model, (vec){-gs, 0.f, fd});
    updateModel();
    esBindRender(current_page+3);

    mIdent(&model);
    mRotY(&model, d2PI+yrot[4]);
    if(selected[current_page+4] == 0.f){mRotZ(&model, t*2.f);}else{mRotZ(&model, selected[current_page+4]);}
    mSetPos(&model, (vec){0.f, 0.f, fd});
    updateModel();
    esBindRender(current_page+4);

    mIdent(&model);
    mRotY(&model, d2PI+yrot[5]);
    if(selected[current_page+5] == 0.f){mRotZ(&model, t*2.f);}else{mRotZ(&model, selected[current_page+5]);}
    mSetPos(&model, (vec){gs, 0.f, fd});
    updateModel();
    esBindRender(current_page+5);

    ///

    mIdent(&model);
    mRotY(&model, d2PI+yrot[6]);
    if(selected[current_page+6] == 0.f){mRotZ(&model, t*2.f);}else{mRotZ(&model, selected[current_page+6]);}
    mSetPos(&model, (vec){-gs, -gs, fd});
    updateModel();
    esBindRender(current_page+6);

    mIdent(&model);
    mRotY(&model, d2PI+yrot[7]);
    if(selected[current_page+7] == 0.f){mRotZ(&model, t*2.f);}else{mRotZ(&model, selected[current_page+7]);}
    mSetPos(&model, (vec){0.f, -gs, fd});
    updateModel();
    esBindRender(current_page+7);

    mIdent(&model);
    mRotY(&model, d2PI+yrot[8]);
    if(selected[current_page+8] == 0.f){mRotZ(&model, t*2.f);}else{mRotZ(&model, selected[current_page+8]);}
    mSetPos(&model, (vec){gs, -gs, fd});
    updateModel();
    esBindRender(current_page+8);

    // display render
    glfwSwapBuffers(window);
}

//*************************************
// input
//*************************************
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if(total_samples < 9){return;}
    if(yoffset > 0.0)
    {
        current_page+=9;
        if(current_page > total_samples){current_page = current_page;} //= current_page-9;
    }
    else
    {
        current_page-=9;
        if(current_page < 0){current_page = 0;}
    }
    updateTitle();
    for(uint i = 0; i < 9; i++){yrot[i] = 0.f;}
}
void copyFile(const char* from, const char* to)
{
    // https://www.cs.toronto.edu/~yuana/ta/csc209/binary-test.c
	FILE * filer, * filew;
	int numr,numw;
    #define BUFFER_LEN 2048
	char buffer[BUFFER_LEN];
	if((filer=fopen(from,"rb"))==NULL){
		fprintf(stderr, "open read file error.\n");
		return;
	}
	if((filew=fopen(to,"wb"))==NULL){
		fprintf(stderr,"open write file error.\n");
        fclose(filer);
		return;
	}
	while(feof(filer)==0){	
	if((numr=fread(buffer,1,BUFFER_LEN,filer))!=BUFFER_LEN){
		if(ferror(filer)!=0){
		fprintf(stderr,"read file error.\n");
        fclose(filer);
	    fclose(filew);
		return;
		}
		else if(feof(filer)!=0);
	}
	if((numw=fwrite(buffer,1,numr,filew))!=numr){
		fprintf(stderr,"write file error.\n");
        fclose(filer);
	    fclose(filew);
		return;
	}
	}
	fclose(filer);
	fclose(filew);
}
void pickModel(const uint id)
{
    selected[id] = t*2.f;
    char cwd[768];
    if(getcwd(cwd, sizeof(cwd)) != NULL)
    {
        char sfrom[2048];
        char sto[2048];
        sprintf(sfrom, "%s/ply/%u.ply", cwd, id+1);
        sprintf(sto, "%s/pick/%u.ply", cwd, id+1);
        puts(sfrom);
        puts(sto);
        copyFile(sfrom, sto);
    }
    printf("Clicked: %u.ply (page: %i)\n", id+1, (id/9)+1);
}
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if(action == GLFW_PRESS)
    {
        if(button == GLFW_MOUSE_BUTTON_LEFT)
        {
            double x, y;
            glfwGetCursorPos(window, &x, &y);
            if(x < (ww*0.5f)-(gs*40.f))
            {
                if(y < (wh*0.5f)-(gs*40.f)){pickModel(current_page);}
                else if(y > (wh*0.5f)+(gs*40.f)){pickModel(6+current_page);}
                else{pickModel(3+current_page);}
            }
            else if(x > (ww*0.5f)+(gs*40.f))
            {
                if(y < (wh*0.5f)-(gs*40.f)){pickModel(2+current_page);}
                else if(y > (wh*0.5f)+(gs*40.f)){pickModel(8+current_page);}
                else{pickModel(5+current_page);}
            }
            else
            {
                if(y < (wh*0.5f)-(gs*40.f)){pickModel(1+current_page);}
                else if(y > (wh*0.5f)+(gs*40.f)){pickModel(7+current_page);}
                else{pickModel(4+current_page);}
            }
        }
        else if(button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            double x, y;
            glfwGetCursorPos(window, &x, &y);
            if(x < (ww*0.5f)-(gs*40.f))
            {
                if(y < (wh*0.5f)-(gs*40.f)){yrot[0] -= 45*DEG2RAD;}
                else if(y > (wh*0.5f)+(gs*40.f)){yrot[6] -= 45*DEG2RAD;}
                else{yrot[3] -= 45*DEG2RAD;}
            }
            else if(x > (ww*0.5f)+(gs*40.f))
            {
                if(y < (wh*0.5f)-(gs*40.f)){yrot[2] -= 45*DEG2RAD;}
                else if(y > (wh*0.5f)+(gs*40.f)){yrot[8] -= 45*DEG2RAD;}
                else{yrot[5] -= 45*DEG2RAD;}
            }
            else
            {
                if(y < (wh*0.5f)-(gs*40.f)){yrot[1] -= 45*DEG2RAD;}
                else if(y > (wh*0.5f)+(gs*40.f)){yrot[7] -= 45*DEG2RAD;}
                else{yrot[4] -= 45*DEG2RAD;}
            }
        }
        else if(button == GLFW_MOUSE_BUTTON_5)
        {
            current_page = total_samples;
            updateTitle();
        }
        else if(button == GLFW_MOUSE_BUTTON_4)
        {
            current_page = 0;
            updateTitle();
        }
    }
}

//*************************************
// process entry point
//*************************************
int main(int argc, char** argv)
{
    // allow custom msaa level
    int msaa = 16;
    if(argc >= 2){msaa = atoi(argv[1]);}

    // help
    printf("----\n");
    printf("James William Fletcher (github.com/mrbid)\n");
    printf("%s - If you chase perfection you just end up making something that is so abstract and tailored to yourself that no one else will understand it.\n", appTitle);
    printf("----\n");
    printf("One command line argument, msaa 0-16\n");
    printf("e.g; ./plv 16\n");
    printf("----\n");
    printf("Left Click = Select\n");
    printf("Right Click = Rotate\n");
    printf("MOUSE4 = Jump to start\n");
    printf("MOUSE5 = Jump to end\n");
    printf("Scroll = Change Page\n");
    printf("----\n");
    printf("regex: \\(page: ([0-9]+)\\)\n");
    printf("----\n");
    printf("Icon by Forest Walter\n");
    printf("https://www.forrestwalter.com/\n");
    printf("----\n");
    printf("%s\n", glfwGetVersionString());
    printf("----\n");

    // init glfw
    if(!glfwInit()){printf("glfwInit() failed.\n"); exit(EXIT_FAILURE);}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_SAMPLES, msaa);
    glfwWindowHint(GLFW_RESIZABLE, 0);
    window = glfwCreateWindow(winw, winh, appTitle, NULL, NULL);
    if(!window)
    {
        printf("glfwCreateWindow() failed.\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    const GLFWvidmode* desktop = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(window, (desktop->width/2)-(winw/2), (desktop->height/2)-(winh/2)); // center window on desktop
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1); // 0 for immediate updates, 1 for updates synchronized with the vertical retrace, -1 for adaptive vsync
    glfwSetWindowIcon(window, 1, &(GLFWimage){16, 16, (unsigned char*)icon_image});

//*************************************
// bind vertex and index buffers
//*************************************
    const time_t st = time(0);
    loadModels("../HeadsNet_ply");
    printf("Time Taken: %lu seconds\n", time(0)-st);

//*************************************
// configure render options
//*************************************
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    //glClearColor(0.3f, 0.745f, 0.8863f, 0.f);
    //glClearColor(0.59608f, 0.37647f, 0.65882f, 0.f);
    makeLambert3();
    shadeLambert3(&position_id, &projection_id, &modelview_id, &lightpos_id, &normal_id, &color_id, &opacity_id);
    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (float*)&projection.m[0][0]);
    glViewport(0, 0, winw, winh);
    ww = (float)winw, wh = (float)winh;
    aspect = ww / wh;
    mIdent(&projection);
    mPerspective(&projection, 55.0f, aspect, 0.01f, FAR_DISTANCE);
    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (float*)&projection.m[0][0]);

//*************************************
// execute update / render loop
//*************************************
    while(!glfwWindowShouldClose(window)){main_loop();}
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
    return 0;
}
