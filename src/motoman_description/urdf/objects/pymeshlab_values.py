import pymeshlab
import sys
import xml.etree.ElementTree as ET
import os
import numpy as np


def compute_values(source_mesh_path,attribute_names,scale_factor,mass):

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(source_mesh_path)
    #scale the mesh to improve the numeric precision
    ms.compute_matrix_from_scaling_or_normalization(axisx=scale_factor, uniformflag=True)
    out_dict = ms.get_geometric_measures()
    volume=out_dict['mesh_volume']
    inertia_tensor = (out_dict['inertia_tensor']*mass)/(volume*(scale_factor**2))
    CoM = out_dict['center_of_mass']/scale_factor
    print(inertia_tensor, "\n", CoM,"\n")
    """convert inertia_tensor and CoM in dictionaries 
    dict.fromkeys(np.reshape(inertia_tensor,9)) creates a new empty dictionary whose keys are the unique values in
    inertia tensor; this is done to preserve the order with the attributes in the xml dictionary because the dict's keys 
    preserve the order from the initial tensor"""
    inertia_values_list = list(dict.fromkeys(np.reshape(inertia_tensor,9)))
    #convert inertia values into string for writing xml values
    inertia_values_as_string = [str(x) for x in inertia_values_list]
    CoM_values_as_string = [str(x) for x in CoM]
    values=inertia_values_as_string + CoM_values_as_string
    values.append(str(mass))
    #convert into tuple to join 
    xml_dictionary = dict(zip(attribute_names, values))
    print(xml_dictionary)
    return xml_dictionary

#xml dictionary contains the values to set; the generic key:value pair are the values to set in the xacro file
#destination-xacro_file is the complete relative path the xacro file
def set_xacro_values(destination_xacro_file,xml_dictionary,name):
    xacro_namespace="http://ros.org/wiki/xacro"
    
    if os.path.exists(destination_xacro_file):
        #if the selected xacro file already exists ask if it was intended to modify it and then modify some values
        value = input("File already exists. Do you want to change it? Y or N\n")
        if value=='Y' or value=='y':
            #this line preserves the namespace when writing the file, must be called before parse
            ET.register_namespace("xacro",xacro_namespace)
            tree = ET.parse(destination_xacro_file)
            root=tree.getroot()
            for attribute in xml_dictionary:
                #search all root node's children which have the name "attribute", that is, whose name matches
                #with the current dictionary key in the cycle
                search_string = ".*[@name='%s']" % (attribute)
                node=root.find(search_string)
                node.attrib['name']=attribute
                node.attrib['value']=xml_dictionary[attribute]
                        
        else:
            return 0
    #else create a new xacro file and set the correct values
    else:
            #root tag
            robot_tag = ET.Element("robot")
            robot_tag.attrib['name']=name
            robot_tag.text="\n"
            #this call goes here in order to avoid the prefix xacro: on the root tag robot
            ET.register_namespace("xacro",xacro_namespace)
            for attribute in xml_dictionary:
                #create one subelement of robot for each property to set
                #ET.QName sets the correct namespace prefix
                property_element=ET.SubElement(robot_tag, ET.QName(xacro_namespace, "property"))
                property_element.attrib['name']=attribute
                property_element.attrib['value']=xml_dictionary[attribute]
                 #add a new line to the end of each tag for a nicer visualization
                property_element.tail='\n'
            tree =ET.ElementTree(robot_tag)
            
            
    tree.write(destination_xacro_file, xml_declaration=True, encoding="utf-8")

           

def main():
    # the program expects the root directory where the mesh is located as the first command line parameter
    source_mesh_directory_path = sys.argv[1]
    

    #first check the arguments are correct
    if (len(sys.argv) == 1) or source_mesh_directory_path == "-h":
        help_message = """USAGE: pymeshlab_values INPUT_MESH_ROOT_FOLDER OUTPUT_FILE_NAME\n\n-INPUT_MESH_FOLDER = root directory where is stored the mesh used for calculations\n-OUTPUT_FILE.xacro.urdf = complete name of the file to write computed values to"""
        print(help_message)
        return -1

    else:
        #check if the mesh file exists
        # and the destination xml/xacro file name as the second parameter
        destination_xacro_file = sys.argv[2]
        #names of the shared xacro attributes
        names = ("Ixx","Ixy", "Ixz", "Iyy", "Iyz","Izz","CoMx","CoMy","CoMz","mass")
        source_mesh_path = os.path.join(source_mesh_directory_path,"texture/textured.dae")
        if os.path.exists(source_mesh_path):
            scale_factor=10
            mass=sys.argv[3]
            # compute geometric values
            xml_dictionary = compute_values(source_mesh_path=source_mesh_path, attribute_names=names,
            scale_factor=scale_factor, mass=float(mass))
            #set those values in the xacro file
            robot_name = source_mesh_directory_path
            set_xacro_values(destination_xacro_file=os.path.join(source_mesh_directory_path,destination_xacro_file),
                xml_dictionary=xml_dictionary,name=robot_name)

        else:
            print("File not found")


main()
