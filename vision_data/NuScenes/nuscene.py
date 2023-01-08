from pyquaternion import Quaternion
from tqdm import tqdm
from loguru import logger
import json
import os, sys
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(
    os.path.join(BASE, 'python-sdk')
)

from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility

class NuScenes:
  """ Database Class for nuScenes to hep query and retrieve information from the database
  - 어쨌든 항상 meta json 데이터에서 token을 기반으로 파일을 찾는 것은 어렵기 때문에 database 접근을 편하게 하기 위한 object를 만든다.

  """
  def __init__(self, version: str = 'v1.0',
               dataroot: str = '/content/drive/MyDrive/internship/MLV Lab/2023 Winter/data/trainval01',
               tableroot: str = '/content/drive/MyDrive/internship/MLV Lab/2023 Winter/data/trainvalmeta/v1.0-trainval',
               verbose: bool = True, 
               map_resolution: float = 0.1):
    self.baseroot = '/content/drive/MyDrive/internship/MLV Lab/2023 Winter/data'
    self.version = version
    self.dataroot = dataroot
    self.tableroot = tableroot
    self.verbose = verbose
    self.table_names = [
        'category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
        'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map'
    ]
    logger.info("INITIALIZING THE TABLE")
    # Explicitly assign tables to help the IDE determine valid class members
    self.category = self.__load_table__('category')
    self.attribute = self.__load_table__('attribute')
    self.visibility = self.__load_table__('visibility')
    self.instance = self.__load_table__('instance')
    self.sensor = self.__load_table__('sensor')
    self.calibrated_sensor = self.__load_table__('calibrated_sensor')
    self.ego_pose = self.__load_table__('ego_pose')
    self.log = self.__load_table__('log')
    self.scene = self.__load_table__('scene')
    self.sample = self.__load_table__('sample')
    self.sample_data = self.__load_table__('sample_data')
    self.sample_annotation = self.__load_table__('sample_annotation')
    self.map = self.__load_table__('map')

    # Initialize the colormap which maps from class names to RGB values
    self.colormap = get_colormap()

    # lidar_tasks = [t for t in ['lidarsef', 'panoptic'] if os.path.exists(os.path.join(self.baseroot, ))]
    # Initialize map mask for each map record
    for map_record in self.map:
      map_record['mask'] = MapMask(os.path.join('/content/drive/MyDrive/internship/MLV Lab/2023 Winter/data/trainvalmeta', 
                                                map_record['filename']), resolution=map_resolution)
    
    if verbose:
      for table in self.table_names:
        logger.info("{} {},".format(len(getattr(self, table)), table))
      logger.info("END INITIALIZING THE TABLE")
    self.__make_reverse_index__(verbose)
  
  def __make_reverse_index__(self, verbose: bool) -> None:
    # Store the mapping from token to table index for each table
    self._token2ind = dict()
    logger.info("REVERSE TABLE")
    for table in self.table_names:
      self._token2ind[table] = dict()
      for ind, member in enumerate(getattr(self, table)):
        self._token2ind[table][member['token']] = ind
    
    # Add shortcut for category name
    for record in self.sample_annotation:
      inst = self.get('instance', record['instance_token'])
      record['category_name'] = self.get('category', inst['category_token'])['name']
    
    # Decorate (adds short-cut) sample_data with sensor information.
    for record in self.sample_data:
      cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
      sensor_record = self.get('sensor', cs_record['sensor_token'])
      record['sensor_modality'] = sensor_record['modality']
      record['channel'] = sensor_record['channel']

   # Reverse-index samples with sample_data and annotations.
    for record in self.sample:
      record['data'] = {}
      record['anns'] = []

    for record in self.sample_data:
      if record['is_key_frame']:
        sample_record = self.get('sample', record['sample_token'])
        sample_record['data'][record['channel']] = record['token']

    for ann_record in self.sample_annotation:
      sample_record = self.get('sample', ann_record['sample_token'])
      sample_record['anns'].append(ann_record['token'])

    # Add reverse indices from log records to map records.
    if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table. This code is not compatible with the teaser dataset.')
    log_to_map = dict()
    for map_record in self.map:
      for log_token in map_record['log_tokens']:
        log_to_map[log_token] = map_record['token']
    for log_record in self.log:
      log_record['map_token'] = log_to_map[log_record['token']]
    logger.info("END REVERSING TABLE")

  def __load_table__(self, table_name):
    with open(os.path.join(self.tableroot, '{}.json'.format(table_name))) as f:
      table = json.load(f)
    return table
  
  def get(self, table_name, token): ## get the actual data based on the index
    return getattr(self, table_name)[self.getind(table_name, token)]
  
  def getind(self, table_name, token): ## get the index from the token on the table
    return self._token2ind[table_name][token]
  
  def get_sample_data(self, sample_data_token, 
                      box_vis_level = BoxVisibility.ANY, 
                      selected_anntokens=None, 
                      use_flat_vehicle_coordinates: bool =False):
    """ Returns the data path as well as all the annotations related to the specific sample data
    :param selected_anntokens: 만약에 지정이 되어 있으면 해당 annotation만을 반환한다.
    :param flat_car: LiDAR 영상을 촬영하는 센서가 차위에 달려 있기 때문에 z 좌표가 xy 평면 바닥에 닿지 않을 수 있다.
    return: (data_path, boxes, camera_intrisic <np.array: 3, 3>)
    """
    sd_record = self.get('sample_data', sample_data_token)
    cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = self.get('sensor', cs_record['sensor_token'])
    pose_record = self.get('ego_pose', sd_record['ego_pose_token'])

    data_path = os.path.join(self.dataroot, self.get('sample_data', sample_data_token)['filename'])

    if sensor_record['modality'] == 'camera':
      cam_intrinsic = np.array(cs_record['camera_intrinsic'])
      imsize = (sd_record['width'], sd_record['height'])
    else:
      cam_intrinsic = None
      imsize = None
    
    # retrieve all sample annotations and map to sensor coordinate system
    if selected_anntokens is not None:
      boxes = list(map(self.get_box, selected_anntokens))
    else:
      boxes = self.get_boxes(sample_data_token)
    
    # Make a list of Box Objects Including the Coordinate System Transforms
    box_list = []
    for box in boxes:
      if use_flat_vehicle_coordinates:
        yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(scalar = np.cos(yaw / 2), vector = [0, 0, np.sin(yaw / 2)]).inverse)
      else:
        ## Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        ##  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

      if sensor_record['modality'] == 'camera' and not \
                    box_in_image(box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):

        continue
      box_list.append(box)
    return data_path, box_list, cam_intrinsic
  
  def get_box(self, sample_annotation_token: str):
    """
    Instantiates a Box class from a sample annotation record.
    :param sample_annotation_token: Unique sample_annotation identifier.
    """
    record = self.get('sample_annotation', sample_annotation_token)
    return Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])
  
  def get_boxes(self, sample_data_token):
    """
    Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
    keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
    sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
    sample_data was captured.
    :param sample_data_token: Unique sample_data identifier.
    """

    # Retrieve sensor & pose records
    sd_record = self.get('sample_data', sample_data_token)
    curr_sample_record = self.get('sample', sd_record['sample_token'])

    if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
      # If no previous annotations available, or if sample_data is keyframe just return the current ones.
      boxes = list(map(self.get_box, curr_sample_record['anns']))

    else:
      prev_sample_record = self.get('sample', curr_sample_record['prev'])

      curr_ann_recs = [self.get('sample_annotation', token) for token in curr_sample_record['anns']]
      prev_ann_recs = [self.get('sample_annotation', token) for token in prev_sample_record['anns']]

      # Maps instance tokens to prev_ann records
      prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

      t0 = prev_sample_record['timestamp']
      t1 = curr_sample_record['timestamp']
      t = sd_record['timestamp']

      # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
      t = max(t0, min(t1, t))

      boxes = []
      for curr_ann_rec in curr_ann_recs:

        if curr_ann_rec['instance_token'] in prev_inst_map:
          # If the annotated instance existed in the previous frame, interpolate center & orientation.
          prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

          # Interpolate center.
          center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                               curr_ann_rec['translation'])]

          # Interpolate orientation.
          rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                                q1=Quaternion(curr_ann_rec['rotation']),
                                                amount=(t - t0) / (t1 - t0))

          box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                              token=curr_ann_rec['token'])
        else:
          # If not, simply grab the current annotation.
          box = self.get_box(curr_ann_rec['token'])

        boxes.append(box)
    return boxes
