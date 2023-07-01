from typing import Any, Callable
import uuid

import sys
sys.path.append(f"{sys.path[0]}/../..")

from lib.Math.math_classes import *
from lib.Exceptions.exceptions import *


class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)


	def add(self, **entries):
		for key, value in entries.items():
			self.__dict__[key] = value


	def add_dict(self, entries: dict):
		return self.add(**entries)


	def dict(self) -> dict:
		return dict(vars(self))


	@staticmethod
	def from_struct(struct: 'Struct'):
		return Struct(**struct.dict())


	@staticmethod
	def from_dict(sdict: dict):
		return Struct(**sdict)


	def __iter__(self):
		return iter(vars(self).values())


	def __len__(self):
		return len(vars(self).keys())


	def __getitem__(self, item):
		if self.__dict__.get(item) is None:
			raise ArgumentsException()
		
		return self.__dict__.get(item)


	def __str__(self, name_obj: str = None):
		string = ", ".join([
			f"{key}: {self.__dict__.get(key)}"
			for key in self.__dict__.keys()
		])
		if name_obj is None:
			return f"Struct({string})"
		else:
			return f"{name_obj}({string})"


	def __repr__(self):
		return self.__str__()


class Ray(Struct):
	"""Basic engine class of ray.
	
	Note:
		Initialize Ray object using ``CoordinateSystem``, ``Point`` and 
		``Vector`` with same dimensions.
	
	"""
	def __init__(self, 
				 cs: CoordinateSystem, 
				 initial_pt: Point, 
				 direction: Vector):

		if (not isinstance(cs, CoordinateSystem) or
				not isinstance(initial_pt, Point) or 
					not isinstance(direction, Vector)):
		
			raise InitializationException()
		
		if cs.init_point.n != initial_pt.n != direction.n:
			raise InitializationException()
		
		self.cs = cs
		self.initial_pt = initial_pt
		self.direction = direction
	

	def normalize(self):
		norm_v = self.direction.normalize()
		return Ray(self.cs, self.initial_pt, norm_v)
	

	def __str__(self):
		return super().__str__("Ray")


class MatrixRay(Matrix):
	 
	def __init__(self, matrix: list[list[Ray]]):

		if not isinstance(matrix, list):
			raise InitializationException()
		
		if not isinstance(matrix[0], list):
			raise InitializationException()
		
		n = len(matrix)
		m = len(matrix[0])
		
		for row in range(n):

			if not isinstance(matrix[row], list):
				raise InitializationException()
			
			if len(matrix[row]) != m:
				raise InitializationException()

			for item in range(m):

				if not isinstance(matrix[row][item], Ray):
					raise InitializationException()

		self.n = n
		self.m = m
		self.matrix = matrix


	def __str__(self) -> str:
		string = "".join(map(str, self.matrix))
		description = f"MatrixRay{self.n}x{self.m}({string})"
		return description


	def __repr__(self) -> str:
		return self.__str__()


class Identifier(Struct):
	"""Basic engine class for identifiers of the objects."""
	identifiers = []

	def __init__(self):
		value = uuid.uuid4()
		Identifier.identifiers.append(value)

		self.value = value 

	
	def __eq__(self, other: "Identifier") -> bool:

		if not isinstance(other, Identifier):
			raise ClassArgumentsException(other)
		
		return self.value == other.value

	@staticmethod
	def __generate__() -> uuid.UUID:
		"""Generating unique identifier for entity.

		Returns:
			uuid.UUID:  Identifier in uuid4 format.

		"""
		value = uuid.uuid4()
		return value
	

	def get_value(self) -> uuid.UUID:
		"""Getting uuid identifier for Identifier object.

		Returns:
			uuid.UUID:  Identifier in uuid4 format.

		"""
		return self.value
	

	def __str__(self):
		return super().__str__("Identifier")
	

class Entity(Struct):
	"""Basic engine class of game entity.

	Note:
		Initialize Entity object using ``CoordinateSystem``.
	
	"""
	def __init__(self, cs: CoordinateSystem):

		if not isinstance(cs, CoordinateSystem):
			raise InitializationException()
		
		self.cs = cs
		self.identifier = Identifier()
	

	def set_property(self, prop: str, value: Any):
		"""Setting any property of entity in any format.

		Args:
			prop (str): Name of property.
			value (Any): Value of property.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.

		"""
		if not isinstance(prop, str):
			raise ClassArgumentsException(prop)

		self.properties.update({prop: value})
	

	def get_property(self, prop: str) -> Any:
		"""Getting any property of entity.

		Args:
			prop (str): Name of property.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.
			ArgumentsException: Wrong argument for method.

		Returns:
			Any: Value of property.

		"""
		if not isinstance(prop, str):
			raise ClassArgumentsException(prop)
		
		if prop not in self.properties.keys():
			raise ArgumentsException()
		
		else:
			return self.properties[prop]
	

	def remove_property(self, prop: str) -> None:
		"""Removing any property of entity.

		Args:
			prop (str): Name of property.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.
			ArgumentsException: Wrong argument for method.

		"""
		if not isinstance(prop, str):
			raise ClassArgumentsException(prop)
		
		if prop not in self.properties.keys():
			raise ArgumentsException()
		
		else:
			self.properties.pop(prop)
		
	
	def __getitem__(self, key: str) -> Any:
		"""Getting property of entity.

		Note:
			To get property use ``EntityObject[prop_name]``.

		Args:
			key (str): Name of property.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.

		Returns:
			Any: Value of property.

		"""
		if not isinstance(key, str):
			raise ClassArgumentsException(key)

		value = self.get_property(key)

		return value


	def __setitem__(self, key: str, value: Any) -> None:
		"""Setting property of entity.

		Note:
			To set property use ``EntityObject[prop_name] = prop_value``.

		Args:
			key (str): Name of property.
			value (Any): Value of property.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.

		"""
		if not isinstance(key, str):
			raise ClassArgumentsException()

		self.set_property(prop=key, value=value)

	
	# def __getattr__(self, name: str) -> Any:
	# 	"""Getting property of entity.

	# 	Note:
	# 		To get property use ``EntityObject.prop_name``.

	# 	Args:
	# 		name (str): Name of property.

	# 	Raises:
	# 		ClassArgumentsException: Wrong type of argument for method.

	# 	Returns:
	# 		Any: Value of property.

	# 	"""
	# 	if not isinstance(name, str):
	# 		raise ClassArgumentsException(name) 
		
	# 	if name in ['cs', 'identifier', 'properties']:
	# 		return super().__getattr__(name)

	# 	else:
	# 		return self[name]


	# def __setattr__(self, name: str, value: Any) -> None:
	# 	"""Setting property of entity.

	# 	Note:
	# 		To set property use ``EntityObject.prop_name = prop_value``.

	# 	Args:
	# 		name (str): Name of property.
	# 		value (Any): Value of property.

	# 	Raises:
	# 		ClassArgumentsException: Wrong type of argument for method.

	# 	"""
	# 	if not isinstance(name, str):
	# 		raise ClassArgumentsException(name) 
		
	# 	if name in ['cs', 'identifier', 'properties']:
	# 		super().__setattr__(name, value)

	# 	else:
	# 		self[name] = value


class EntitiesList:
	"""Basic engine class of list of game entities.
	
	Note:
		Initialize ``EntitiesList`` object using list of ``Entity``'s. 
		Defaults to None.

	"""
	def __init__(self, entities: list[Entity] = []):

		if not isinstance(entities, list):
			raise InitializationException()
		
		for entity in entities:

			if not isinstance(entity, Entity):
				raise InitializationException()
		
		self.entities = entities

	
	def __getitem__(self, identifier: Identifier) -> Entity:
		"""Getting entity object by identifier.

		Note:
			To get entity object use ``EntityListObject[identifier]``.

		Args:
			identifier (Identifier): Identifier of entity object.
		
		Raises:
			ClassArgumentsException: Wrong type of argument for method.

		Returns:
			Entity: Entity object.

		"""
		if not isinstance(identifier, Identifier):
			raise ClassArgumentsException(identifier)
		
		for entity in self.entities:
			if entity.identifier == identifier:
				return entity



	def append(self, entity: Entity) -> None:
		"""Adding entity to list.

		Args:
			entity (Entity): Antity to add.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.
			ArgumentsException: Wrong argument for method.

		"""
		if not isinstance(entity, Entity):
			raise ClassArgumentsException(entity)

		if entity.identifier in [i.identifier for i in self.entities]:
			raise ArgumentsException 
		
		self.entities.append(entity)
		

	def remove(self, entity: Entity) -> None:
		"""Removing entity from list.

		Args:
			entity (Entity): Entity to remove.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.
			ArgumentsException: Wrong argument for method.

		"""
		if not isinstance(entity, Entity):
			raise ClassArgumentsException(entity)
		
		if entity not in self.entities:
			raise ArgumentsException()
		
		else:
			self.entities.remove(entity)
	
	
	def _get_all_identifiers(self) -> list[Identifier]:

		list_id = []

		for entity in self.entities:
			list_id.append(entity.identifier)
		
		return list_id

	
	def get(self, identifier: Identifier) -> Entity:
		"""Getting entity object by identifier.

		Args:
			identifier (Identifier): Identifier of entity object.
		
		Raises:
			ClassArgumentsException: Wrong type of argument for method.

		Returns:
			Entity: Entity object.

		"""
		if not isinstance(identifier, Identifier):
			raise ClassArgumentsException(identifier)
		
		if identifier not in self._get_all_identifiers():
			raise ArgumentsException()
		
		else:
		
			for entity in self.entities:
				if entity.identifier == identifier:
					return entity
		
	
	def execute(self, func: Callable[[Entity], Entity]) -> None:
		"""Exexcuting given function on any entity in list.

		Args:
			func (Callable[[Entity], Entity]): Function with one ``Entity``
				object argument, which returns ``Entity`` object.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.
			ArgumentsException: Wrong argument for method.

		"""

		if not callable(func):
			raise ArgumentsException()

		changed_entities = list(map(func, self.entities))
		
		for entity in changed_entities:

			if not isinstance(entity, Entity):
				raise ClassArgumentsException(entity)
		
		self.entities = changed_entities


class Object(Entity):
	"""Basic class of engine object based on Entity.

	Note:
		Initialize Object object using ``CoordinateSystem``, ``Point`` and 
		``Vector`` with same dimensions.
	
	"""
	def __init__(self,
				cs: CoordinateSystem, 
				position: Point, 
				direction: Vector):
				
		if not isinstance(position, Point) or \
			not isinstance(direction, Vector) or \
			not isinstance(cs, CoordinateSystem): 
				raise InitializationException()
		
		# vector = direction.normalize()

		super().__init__(cs)
		self.position = position
		self.direction = direction

	
	def move(self, direction: Vector):
		"""Moving object in coordinate system without rotating.

		Args:
			direction (Vector): Vector to move object.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.

		"""
		if not isinstance(direction, Vector):
			raise ClassArgumentsException()
		
		self.position = self.position.addition(direction)

	
	def planar_rotate(self, inds: tuple[int, int], angle: float, n: int):
		"""Rotating object in plain given angle.

		Args:
			inds (tuple[int, int]): Two axis giving plain to rotate in.
			angle (float): Angle in radians
			n (int): Dimension of coordinate system.

		"""
		self.direction = Matrix.rotation_matrix(inds, angle, n) * \
							Matrix(self.direction.matrix)
		
	
	def rotate_3d(self, *angles):
		"""Rotating object in 3d using Tait-Bryan matrixes. Based on 
		``Vector.rotate()`` method.
		
		"""
		self.direction = self.direction.rotate(angles)

	
	def set_position(self, position: Point):
		"""Setting position of object.

		Args:
			position (Point): Position to set.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.
		"""

		if not isinstance(position, Point):
			raise ClassArgumentsException()
		
		self.set_property('position', position)

	
	def set_direction(self, direction: Vector):
		"""Setting direction of object.

		Args:
			direction (Vector): Direction to set.

		Raises:
			ClassArgumentsException: Wrong type of argument for method.

		"""
		if not isinstance(direction, Vector):
			raise ClassArgumentsException()
		
		self.set_property('direction', direction.normalize())
	
	#TODO
	def intersection_distance(self, ray: Ray) -> float:
		pass


class Camera(Object):
	"""Basic class of engine camera based on Object.

	Note:
		Initialize Camera object using ``CoordinateSystem``, position as ``Point``, 
		direction as ``Vector``, fov as ``float``, draw distance as ``float``,
		vfov as ``float`` and point to look at as ``Point``. Defaults of vfov 
		and point to look at to None.
	
	"""
	def __init__(self, 
				 cs: CoordinateSystem,
				 position: Point,
				 direction_or_lookat: Union[Vector, Point],
				 fov: float,
				 draw_distance: float, 
				 vfov: float = None):
	
		if not isinstance(cs, CoordinateSystem) or \
			not isinstance(position, Point) or \
			not isinstance(direction_or_lookat, Vector) or \
			not isinstance(draw_distance, (int,float)) or \
			not isinstance(fov, float):
				raise InitializationException()

		if (vfov is not None) and not isinstance(vfov, float):
			raise InitializationException()
		
		if vfov is None:
			self.vfov = round(math.atan(16/9 * math.tan(fov/2)), PRECISION)
		
		else:
			self.vfov = vfov
		
		if isinstance(direction_or_lookat, Point):
			look_at = direction_or_lookat
			v1 = Vector(look_at.matrix)
			v2 = Vector(position.matrix)
			direction_or_lookat = v1 - v2
		
		self.fov = fov
		self.draw_distance = draw_distance 

		super().__init__(cs, position, direction_or_lookat)
	

	def get_rays_matrix(self, n: int, m: int) -> MatrixRay:

		a, b = self.fov, self.vfov
		
		#Or swap n and m places
		delta_a = a / (m-1)
		delta_b = b / (n-1)

		a_list = [0 for _ in range(m)]
		b_list = [0 for _ in range(n)]

		for i in range(m):
			a_list[i] = delta_a * i - (a / 2)
		
		for j in range(n):
			b_list[j] = delta_b * j - (b / 2)

		ray_matrix = [[0 for _ in range(m)] for _ in range(n)]

		for i in range(n):
			for j in range(m):
				v = Vector(self.direction.matrix)

				result = Matrix.rotation_matrix((1, 0), b_list[i]) * \
						 Matrix.rotation_matrix((2, 0), a_list[j]) * \
						 Matrix(v.matrix)
				
				v_ij = Vector(result.matrix)
				v_ij = (v.length())**2 / (v % v_ij) * v_ij
				init_pt = self.position + v_ij

				ray = Ray(self.cs, init_pt, v).normalize()
				ray_matrix[i][j] = ray
		
		return MatrixRay(ray_matrix)
	

class HyperPlane(Object):

	def __init__(self,
					cs: CoordinateSystem, 
					position: Point, 
					normal: Vector):

		if not isinstance(position, Point) or \
			not isinstance(normal, Vector) or \
			not isinstance(cs, CoordinateSystem): 
				raise InitializationException()
		
		super().__init__(cs, position, normal)
	

	def intersection_distance(self, ray: Ray) -> float:
		if not isinstance(ray, Ray):
			raise ArgumentsException()
		
		n = self.direction
		r = ray.direction
		x_1 = Vector(ray.initial_pt.matrix)
		x_0 = Vector(self.position.matrix)

		if abs(n % r) < 10**(-PRECISION):

			if abs(n % (x_1 - x_0)) < 10**(-PRECISION):
				return 0
			else:
				return -1
			
		else:

			nakl = x_1 - x_0
			t = -1 * (n % (nakl)) / (n % r)

			if t < 0:
				return -1
			else:
				return t
			
	#TODO
	def planar_rotate(self, inds: tuple[int, int], angle: float, n: int):
		return super().planar_rotate(inds, angle, n)
	
	
	#TODO
	def rotate_3d(self, *angles):
		return super().rotate_3d(*angles)


class HyperEllipsoid(Object):

	def __init__(self,
					cs: CoordinateSystem, 
					position: Point, 
					direction: Vector,
					semiaxes: list[float]):

		if not isinstance(position, Point) or \
			not isinstance(direction, Vector) or \
			not isinstance(cs, CoordinateSystem) or \
			not isinstance(semiaxes, list): 
				raise InitializationException()
		
		for axis in semiaxes:
			if not isinstance(axis, (int,float)):
				raise InitializationException()
		
		super().__init__(cs, position, direction)

		self.semiaxes = semiaxes
	
	#TODO
	def planar_rotate(self, inds: tuple[int, int], angle: float, n: int):
		pass
	
	#TODO
	def rotate_3d(self, *angles):
		pass

	
	def intersection_distance(self, ray: Ray) -> float:
		a, b, c = self.semiaxes[0], self.semiaxes[1], self.semiaxes[2]
		x_e, y_e, z_e = self.position[0], self.position[1], self.position[2]
		x_p, y_p, z_p = ray.initial_pt[0], ray.initial_pt[1], ray.initial_pt[2]
		r_x, r_y, r_z = ray.direction[0], ray.direction[1], ray.direction[2]
		m, n, k = x_p - x_e, y_p - y_e, z_p - z_e

		# At**2 + Bt + C = 0
		A = r_x**2 * b**2 * c**2 + \
			r_y**2 * a**2 * c**2 + \
			r_z**2 * b**2 * a**2
		B = 2 * r_x * b**2 * c**2 * m + \
			2 * r_y * a**2 * c**2 * n + \
			2 * r_z * b**2 * a**2 * k
		C = m**2 * b**2 * c**2 + \
			n**2 * a**2 * c**2 + \
			k**2 * b**2 * a**2  - a**2 * b**2 * c**2
	
		D = B**2 - 4*A*C

		if abs(D) < 10**(-PRECISION):
			return round(-B / (2 * A), PRECISION)
		
		if D < 0:
			return -1
		
		if D > 0:
			root_1 = round((-B + D**0.5) / (2 * A), PRECISION)
			root_2 = round((-B - D**0.5) / (2 * A), PRECISION)

			if root_1 < 0 and root_2 < 0:
				raise EngineException()
			
			if min(root_1, root_2) > 0:
				return min(root_1, root_2)
			
			else:
				return max(root_1, root_2)


class Canvas(Struct):

	def __init__(self,
				 cs: CoordinateSystem, 
				 entity_list: EntitiesList, 
				 n: int,
				 m: int):
	
		if not isinstance(entity_list, EntitiesList) or \
			not isinstance(cs, CoordinateSystem) or \
			not isinstance(n, int) or \
			not isinstance(m, int): 
				raise InitializationException()
		
		self.cs = cs
		self.entity_list = entity_list
		self.n = n
		self.m = m
		#TODO Matrix of valid distances
		self.distances = Matrix.zero(n, m)
		self.closest = 0
		self.farther = 0


	def draw(self) -> None:
		charmap = ".:;><+r*zsvfwqkP694VOGbUAKXH8RD#$B0MNWQ%&@"
		
		step = (self.farther - self.closest) / (len(charmap))
		print("-"*(self.m + 2))
		
		for i in range(self.n):
			line_list = [" " for _ in range(self.m)]
			for j in range(self.m):

				if abs(step) < 10**(-PRECISION):
					line_list[j] = charmap[-1]
					continue

				if self.distances[i, j] != -1:
					index = int((self.distances[i, j] - self.closest) / step)
					
					if index >= 42:
						index = 41

					line_list[j] = charmap[41-index]

			line_list.insert(0, "|")
			line_list.append("|")
			line_str = "".join(line_list)
			
			print(line_str, sep="")
		print("-"*(self.m + 2))


	def draw_curse(self) -> list[str]:
		charmap = ".:;><+r*zsvfwqkP694VOGbUAKXH8RD#$B0MNWQ%&@"
		
		step = (self.farther - self.closest) / (len(charmap))
		lines = ["" for _ in range(self.n)]
		for i in range(self.n):
			line_list = [" " for _ in range(self.m)]
			for j in range(self.m):

				if abs(step) < 10**(-PRECISION):
					line_list[j] = charmap[-1]
					continue

				if self.distances[i, j] != -1:
					index = int((self.distances[i, j] - self.closest) / step)
					
					if index >= 42:
						index = 41

					line_list[j] = charmap[41-index]

			line_str = "".join(line_list)
			
			lines[i] = line_str
		
		return lines

	def update(self, camera: Camera):

		if not isinstance(camera, Camera):
			raise ArgumentsException()
		
		matrix_ray = camera.get_rays_matrix(self.n, self.m)	
		min_dist = camera.draw_distance + 1
		max_dist = -1

		for i in range(self.n):
			for j in range(self.m):

				ray = matrix_ray[i, j]
				min_local = camera.draw_distance + 1
				
				for entity in self.entity_list.entities:

					if not isinstance(entity, (HyperEllipsoid, HyperPlane)) :
						continue

					dist = entity.intersection_distance(ray)

					if min_local > dist and dist >= 0:
						min_local = dist

					if min_dist > dist and dist >= 0:
						min_dist = dist
						self.closest = dist
					
					if max_dist < dist and dist >= 0:
						max_dist = dist
						self.farther = dist
				
				if min_local > camera.draw_distance:
					result_dist = -1
				else:
					result_dist = min_local

				self.distances[i, j] = result_dist


class EventSystem:

	def __init__(self):
		self.events = {}


	def add(self, name: str):
		self.events.update({name: []})


	def remove(self, name: str):
		if name not in self.events.keys():
			raise ArgumentsException()

		self.events.pop(name)


	def add_handle(self, name: str, func: Callable):
		if name not in self.events.keys():
			raise ArgumentsException()
		
		func_list = self.events.get(name)
		func_list.append(func)
		self.events.update({name: func_list})


	def remove_handle(self, name: str, func: Callable):
		if name not in self.events.keys():
			raise ArgumentsException()
		
		func_list = self.events.get(name)
		func_list.remove(func)
		self.events.update({name: func_list})


	def get_handle(self, name: str):
		if name not in self.events.keys():
			raise ArgumentsException()
		
		func_list = self.events.get(name)
		return func_list


	def trigger(self, name: str, *args):
		if name not in self.events.keys():
			raise ArgumentsException()
		
		func_list = self.events.get(name)

		for func in func_list:
			func(*args)
	
	def __getitem__(self, name: str):
		return self.get_handle(name)


class Configuration:

	def __init__(self, filepath: str = ""):
		if not isinstance(filepath, str): 
			raise InitializationException()
		
		self.filepath = filepath
		self.configuraton = {}
	
	def set_variable(self, var: str, value: Union[Any, None]):
		if not isinstance(var, str): 
			raise ArgumentsException()
		
		self.configuraton.update({var: value})
	
	def get_variable(self, var: str):
		if not isinstance(var, str): 
			raise ArgumentsException()
		
		return self.configuraton.get(var)

	
class Game:
	"""Basic class of object, which creates main game.
	
	Note:
		Initialize ``Game`` object using ``CoordinateSystem`` and 
		``EntitiesList``.

	"""
	def __init__(self, 
	      		 cs: CoordinateSystem,
			     es: EventSystem, 
				 entities: EntitiesList):

		if not isinstance(cs, CoordinateSystem): 
			raise InitializationException()
		
		if not isinstance(es, EventSystem):
			raise InitializationException()
		
		if not isinstance(entities, EntitiesList):
			raise InitializationException()
		
		self.cs = cs
		self.es = es
		self.entities = entities
		

	def run() -> None:
		"""Method for running game.

		Todo:
			Method to realize.
		
		"""
		pass


	def update() -> None:
		"""Method for updating state of game.

		Todo:
			Method to realize.
			
		"""
		pass


	def exit() -> None:
		"""Method for exiting game session.
		
		Todo:
			Method to realize.

		"""
		pass


	def get_event_system(self):
		return self.es
	

	def apply_configuration(self):
		pass


	def get_entity_class(self):
		"""Creating Game.Entity class with fixed coordinate system from current Game
		object.

		Returns:
			class: Entity class of current Game.

		"""
		class GameEntity(Entity):

			def __init__(pself):
				super().__init__(self.cs)

		return GameEntity


	def get_ray_class(self):
		"""Creating Game.Ray class with fixed coordinate system from current Game
		object.

		Returns:
			class: Ray class of current Game.

		"""
		class GameRay(Ray):

			def __init__(pself, initial_pt: Point, direction: Vector):
				super().__init__(self.cs, initial_pt, direction)

		return GameRay
	

	def get_object_class(self):
		"""Creating Game.Object class, inherited from Entity class, with fixed 
		coordinate system from current Game object.

		Returns:
			class: Object class of current Game.

		"""
		class GameObject(Object):

			def __init__(pself, position: Point, direction: Vector):
				super().__init__(self.cs, position, direction)
				
		return GameObject


	def get_camera_class(self):
		"""Creating ``Game.Camera`` class, inherited from Object class, with 
		fixed coordinate system from current Game object.

		Returns:
			class: Camera class of current Game.

		"""
		class GameCamera(Camera):

			def __init__(pself, 
						 position: Point,
						 direction: Vector,
						 fov: float,
						 draw_distance: float, 
						 vfov: float = None, 
						 look_at: Point = None):
				
				super().__init__(self.cs,
								 position,
								 direction,
								 fov,
								 draw_distance, 
								 vfov, 
								 look_at)

		return GameCamera
	

	def get_hyperplane_class(self):
		"""Creating ``Game.HyperPlane`` class, inherited from Object class, with 
		fixed coordinate system from current Game object.

		Returns:
			class: HyperPlane class of current Game.

		"""
		class GameHyperPlane(HyperPlane):

			def __init__(pself, 
						 position: Point, 
						 normal: Vector):
				super().__init__(self.cs, position, normal)

		return GameHyperPlane
	

	def get_hyperellipsoid_class(self):
		"""Creating ``Game.HyperEllipsoid`` class, inherited from Object class,
		with fixed coordinate system from current Game object.

		Returns:
			class: HyperEllipsoid class of current Game.

		"""
		class GameHyperEllipsoid(HyperEllipsoid):

			def __init__(pself,
						 position: Point, 
						 direction: Vector, 
						 semiaxes: list[float]):
				
				super().__init__(self.cs, position, direction, semiaxes)

		return GameHyperEllipsoid
	

	def get_canvas_class(self):
		"""Creating ``Game.Canvas`` class
		with fixed coordinate system and list of entities from current Game 
		object.

		Returns:
			class: Canvas class of current Game.

		"""
		class GameCanvas(Canvas):

			def __init__(pself, 
						 n: int, 
						 m: int):
				
				super().__init__(self.cs, self.entity_list, n, m)

		return GameCanvas


def camera_body_move_handler(camera: Camera, key_on_board: int):

	if not isinstance(camera, Camera) or not isinstance(key_on_board, int):
		raise ArgumentsException()

	if key_on_board == 261:
		camera.position = camera.position + Vector(0, 0, 1)
	
	if key_on_board == 259:
		camera.position = camera.position + Vector(1, 0, 0)

	if key_on_board == 260:
		camera.position = camera.position + Vector(0, 0, -1)

	if key_on_board == 258:
		camera.position = camera.position + Vector(-1, 0, 0)


def camera_head_move_handler(camera: Camera, key_on_board: int):

	if not isinstance(camera, Camera) or not isinstance(key_on_board, int):
		raise ArgumentsException()

	if key_on_board == ord('a'):
		v_matrix = Matrix.rotation_matrix((2, 0), -1 * camera.fov / 10) * \
							Matrix(camera.direction.matrix)
		
		

		camera.direction = Vector(v_matrix.matrix)
	
	if key_on_board == ord('d'):
		v_matrix = Matrix.rotation_matrix((2, 0), camera.fov / 10) * \
							Matrix(camera.direction.matrix)

		camera.direction = Vector(v_matrix.matrix)
	
	if key_on_board == ord('w'):
		v_matrix = Matrix.rotation_matrix((1, 0), camera.vfov / 10) * \
							Matrix(camera.direction.matrix)

		camera.direction = Vector(v_matrix.matrix)
	
	if key_on_board == ord('s'):
		v_matrix = Matrix.rotation_matrix((1, 0), -1 * camera.vfov / 10) * \
							Matrix(camera.direction.matrix)

		camera.direction = Vector(v_matrix.matrix)


if __name__ == "__main__":
	
	# initialize game
	vs = VectorSpace([Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)])
	cs =  CoordinateSystem(vs, Point(0, 0, 0))
	es = EventSystem()
	elip = HyperEllipsoid(cs, Point(20, 0, 0), Vector(1, 0, 0), [6, 8, 10])
	plane = HyperPlane(cs, Point(40, 0, 0), Vector(1.0, 0.5, 0.0))
	ent_list = EntitiesList([plane, elip])
	camera = Camera(cs, Point(0, 0, 0), Vector(10, 0, 0), math.pi / 2, 60, math.pi / 2)
	G = Game(cs, es, ent_list)
	GC = Canvas(cs, ent_list, 10, 30)
	GC.update(camera)

	# init event system
	G.es.add("OnCameraMove")
	G.es.add_handle("OnCameraMove", camera_body_move_handler)
	

