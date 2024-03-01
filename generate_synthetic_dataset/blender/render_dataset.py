import bpy
import os
import math
import random
import decimal
import numpy as np
import sys
import decimal
from mathutils import Vector
import json

bpy.context.scene.render.engine = 'CYCLES' # use Cycles render engine 
for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'

## clear scene ##
# https://b3d.interplanety.org/en/how-to-delete-object-from-scene-through-the-blender-python-api/



def decnum(Lbound,Ubound):
    N = float(decimal.Decimal(random.randrange(int(Lbound*100), int(Ubound*100)))/100) 
    print(N)
    return N

# check below if everything is really needed

def bake_phys():  # bake physics
    for scene in bpy.data.scenes:
        bpy.ops.ptcache.free_bake_all()
        bpy.ops.ptcache.bake_all(bake=True)
        



def makevar(): # make dictionary including all variables from txt
    dict_vars = {}
    for line in text:                   # read all lines
        if line[0] != "#":
            if "=" in line:             # then make var
                arr = line.split("=")   # after : is the value
                varname = arr[0]
                val = arr[1]
                if "#" in val:          # don't look at comment
                    val = val.replace('\t','')
                val = val.split("#")
                value = val[0].strip()
                dict_vars[varname] = value.replace('\n','')
    return dict_vars


def makeparent(child,parent):
    bpy.ops.object.select_all(action='DESELECT')        # deselect everything 
    child.select_set(True)                              # select child
    parent.select_set(True)                             # select parent
    bpy.context.view_layer.objects.active = parent      # active object will be the parent
    bpy.ops.object.parent_set(type='VERTEX')
    
    bpy.ops.object.select_all(action='DESELECT') # deselect everything 

    child.location = parent.location
    child.rotation_euler = parent.rotation_euler


def save_json(out_dict): # https://blender.stackexchange.com/questions/237825/export-json-files-with-properties
    n = out_dict["Imnum"]
    jsonname = str(n) + '.json'
    json_object = json.dumps(out_dict, indent=4)
    # using "x" instead of "w" to error if the file already exists, though very unlikely due to line 29
    print(out_path + jsonname)
    with open(out_path + jsonname, "w") as outfile:
        outfile.write(json_object)


#obtain settings from settingsfile
settingsfile = r"./render_settings.txt"
with open(settingsfile) as f:
    text = f.readlines()
    set_vars = makevar()
    
    
scenes = int(set_vars['num_scenes'])
n_obj_l = int(set_vars['num_obj_l'])
n_obj_h = int(set_vars['num_obj_h'])

n_cam_l = int(set_vars['num_cam_l'])
n_cam_h = int(set_vars['num_cam_h'])


n_lights_l = int(set_vars['num_lights_l'])
n_lights_h = int(set_vars['num_lights_h'])

path = set_vars['path']
file_name1 = set_vars['file_name1']
file_name2 = set_vars['file_name2']
inner = set_vars['inner']
shape = set_vars['shape']


bake = int(set_vars['bake'])

start_frame = int(set_vars['start_frame'])

out_path = set_vars['out_path']

noise = int(set_vars['noise'])

file_path1 = path + file_name1
file_path2 = path + file_name2

## below lets us enter any number of images in the settingsfile
tex_im = []
for entry in set_vars: 
    print(entry[0:3])
    if entry[0:3] == 'tex':
        tex_im.append(set_vars[entry])
        

output = []



# set the path for all file output nodes:
for scene in bpy.data.scenes:
    for node in scene.node_tree.nodes:
        if node.type == 'OUTPUT_FILE':
            print('OUTPATH', out_path)
            node.base_path = out_path
 

fn = start_frame


for p in range(scenes):
    # select all objects
    bpy.ops.object.select_all(action='SELECT')
    # delete all selected objects
    bpy.ops.object.delete()

    num = random.randrange(n_obj_l,n_obj_h)
    
    for i in range(num):
        file_path = random.choice([file_path1,file_path2])    # use one of the two models
        obj = bpy.ops.wm.append(
            filepath=os.path.join(file_path, inner, shape),
            directory=os.path.join(file_path, inner),
            filename=shape
            )
    
        # recalculate normals
    for obj in bpy.context.selected_objects[:]:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # go edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        # select al faces
        bpy.ops.mesh.select_all(action='SELECT')
        # recalculate outside normals 
        bpy.ops.mesh.normals_make_consistent(inside=False)
        # go object mode again
        bpy.ops.object.editmode_toggle()


    xlim = 2.0
    ylim = 2.0
    zlim = 2.0

    n = 1
    oi = 100
    k = 1
    obj_names = []
    objects = []
    empty_names = []
    

    for objn in bpy.context.scene.objects:

        if objn.name == 'Plane':  # change texture of plane

            for node in objn.data.materials[0].node_tree.nodes: ##
                if node.type == 'TEX_NOISE': # add random noise texture to plane
                    node_val = random.choice([decnum(0,1e2),decnum(0,1e4),decnum(0,1e6)])
                    node.inputs[2].default_value = node_val
                elif node.type == 'RGB': # change color of plane
                    node.outputs[0].default_value = (decnum(0,1),decnum(0,1),decnum(0,1),1)
        elif objn.type == 'MESH' and objn.data.name[0] == shape[0]:
            # assign name
            name = "fillet" + str(n)
            
            objn.name = name
            objn.data.name = name
            objects.append(name)
            obj_names.append(name)
            
            obji = bpy.data.objects[name] # assign pass index
            obji.pass_index = oi
            oi += 10
            
            
            # move object in XY plane
            x = decnum(-xlim,xlim)
            y = decnum(-ylim,ylim)
            
            zstep = 1
            z = decnum((k+1)*zstep,(k+2)*zstep)
            k += 2   
            
            rotx = random.choice([decnum(-5,5),decnum(175,185)])  # degrees
            roty = decnum(-5,5)
            rotz = decnum(0,360)

            obj = bpy.context.blend_data.objects[name]
            new_loc = [x,y,z] # randomize this?

            obj.location = new_loc  # relocate object
            
            # resize object
            scale = decnum(1,1.2)          
            old_sc = obj.scale 
            new_sc = old_sc * scale
            obj.scale = new_sc
            
            # rotate object
            pi = math.pi
            new_rot = (rotx*pi/180,roty*pi/180,rotz*pi/180)
            obj.rotation_euler = new_rot
            
            
            # add emptys, make objects parents
            ename = "empty" + str(n)
            empty_names.append(ename) # for later use

            e = bpy.data.objects.new(ename, None)  # add empty
            bpy.context.scene.collection.objects.link(e) # link to scene
  
            n += 1 # update obj number


    if noise:
        for num_noise in range(random.randrange(0,10)):  # add random objects
            type = random.randrange(0,4)
            if type == 0:   
                bpy.ops.mesh.primitive_cylinder_add()
            elif type == 1:
                bpy.ops.mesh.primitive_cube_add()
            elif type == 2:
                bpy.ops.mesh.primitive_cone_add()
            elif type == 3:
                bpy.ops.mesh.primitive_ico_sphere_add()
            # created object is active object
            obj = bpy.context.active_object
            bpy.ops.object.modifier_add(type='COLLISION') ## add collision modifier 
            bpy.ops.rigidbody.object_add()#add rigidbody modifier
            
            # create random texture to object
            mat = bpy.data.materials.new(name="random_mat")
            mat.use_nodes = True
           
            # give object random color
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (decnum(0,1),decnum(0,1),decnum(0,1),1)
            
            for i in np.random.choice(21,5):
                if i != 19 and i != 3 and i != 0 and i != 2: # skip all 3 dims
                    mat.node_tree.nodes["Principled BSDF"].inputs[i].default_value = decnum(0,1)

            obj.active_material = mat
 
            # now move and scale the active object
            # move object in XY plane
            x = random.choice([decnum(-8,-3),decnum(3,8)])
            y = random.choice([decnum(-8,-3),decnum(3,8)])
            
            z = 3
            
            rotx = decnum(0,180)  # degrees
            roty = decnum(0,180)
            rotz = decnum(0,180)

            new_loc = [x,y,z] # randomize this?

            obj.location = new_loc  # relocate object
            
            # resize object
            scale = decnum(1,2)

            old_sc = obj.scale 
            new_sc = old_sc * scale
            obj.scale = new_sc
            
            # rotate object
            pi = math.pi
            new_rot = (rotx*pi/180,roty*pi/180,rotz*pi/180)
            obj.rotation_euler = new_rot
        
    # run animation, then select scene to render
    # only useful if physics is simulated
    if bake == 1:
        bake_phys()  # bake all physics
        bpy.context.scene.frame_set(fn)
    else:
        bpy.context.scene.frame_set(1)
 
    save_rgb = [] # to save light colors
    n_lights = random.randrange(n_lights_l,n_lights_h)   
    for l in range(n_lights):

        # create light datablock, set attributes
        light_data = bpy.data.lights.new(name="light", type='POINT')
        light_data.energy = decnum(1100,1300)# decnum(2,3)/n_lights # this is for sun
        R = decnum(0,1)
        G = decnum(0,1)
        B = decnum(0,1)
        save_rgb.append([R,G,B])
        light_data.color = (R,G,B)#(decnum(0,1),decnum(0,1),decnum(0,1))

        # create new object with our light datablock
        light_object = bpy.data.objects.new(name="light", object_data=light_data)

        # link light object
        bpy.context.collection.objects.link(light_object)

        # make it active 
        bpy.context.view_layer.objects.active = light_object

        xl = decnum(-6,6) 
        yl = decnum(-6,6) 
        zl = decnum(4,6)
        
        #change location
        light_object.location = (xl, yl, zl) ## randomize light location?

        # update scene, if needed
        dg = bpy.context.evaluated_depsgraph_get() 
        dg.update()
    cn = random.randrange(n_cam_l,n_cam_h) # number of cameras for this scene
    
    # obtain object type and locations
    obj_loc = []
    obj_rot = []
    obj_type = []
    
    for name, ename in zip(obj_names,empty_names):
        print([name,ename])
        obj = bpy.data.objects[name]
        print(obj.matrix_world.translation)
        obj_loc.append(obj.matrix_world.translation)

        # get updated location of emptys
        obj_empty = bpy.data.objects[ename]

        # obtain location of object center after dynamics 
        print(obj_empty.matrix_world) 
        transl_empty = obj_empty.matrix_world.translation
        print('T_'+ename, transl_empty)
  
  
    for a in range(cn + 1):
        bpy.ops.object.select_all(action='DESELECT')
        print('A',a)
        
        # add cam
        lim = 20 # limit of camera pos on axis
        look_at = e # latest empty
        
        Z = decnum(int(lim/4),int(lim*2))
      
        X = decnum(-lim,lim)
        Y = decnum(-lim,lim)
          
        # https://blender.stackexchange.com/questions/176296/add-camera-at-random-position-through-python
        # add cameras
        bpy.ops.object.camera_add(align='WORLD',location = (X,Y,Z)) # add camera

        camera = bpy.data.objects['Camera']  # access camera
        print('OBJECT NAMES',obj_names)
 

        z = camera.location[2]*decnum(0.8,1.6)
        radius = Vector((camera.location[0], camera.location[1], 0)).length
        angle = 2 * math.pi * random.random()

        # Randomly place the camera on a circle around the object at the same height as the main camera
        new_camera_pos = Vector((radius * math.cos(angle), radius * math.sin(angle), z))

        bpy.ops.object.camera_add(enter_editmode=False, location=new_camera_pos)

        # Add a new track to constraint and set it to track your object
        track_to = bpy.context.object.constraints.new('TRACK_TO')
        track_to.target = look_at
        track_to.track_axis = 'TRACK_NEGATIVE_Z'
        track_to.up_axis = 'UP_Y'

        # Set the new camera as active
        bpy.context.scene.camera = bpy.context.object

      
        output.append('Scene: ' + str(p) + ', frame: ' + str(fn) + '. Camera: ' + str(a))# + str(checkmark)) 
        bpy.context.scene.frame_set(fn)
        
        # create (masked) data
        # make sure 'Object Index' is turned on in Layer Properties!
        bpy.ops.render.render(write_still=True)    # write scene to image taken by current camera
        
        '''define dict'''
        if len(str(fn))<4:
          imnum = '0'+str(fn)
        else:
          imnum = str(fn)
        
        savevar = {"Imnum": imnum,  
                  "Imname": "Image"+imnum+".png",
                  "Maskname": "Mask"+imnum+".png",
                  "Scene": p,
                  "# Objects": num, 
                  "# Lights": n_lights,
                  "Light colors":  str(save_rgb),
                  "Cam. Location": str(new_camera_pos)
                  }
        # write .json
        save_json(savevar)
        
        fn += 1  # increase frame number


