import rasterio
from geopy.geocoders import Nominatim
from rasterio.warp import transform


def get_lat_lon(tiff_path, x, y):
    """
    Получение широты и долготы из координат пикселя в GeoTIFF.
    """
    try:
        with rasterio.open(tiff_path) as src:
            # Получение координат в CRS изображения
            # rasterio использует (row, col), что эквивалентно (y, x)
            xs, ys = src.xy(y, x)

            # Проверка CRS
            if src.crs.to_epsg() == 4326:
                return ys, xs

            # Если не 4326, выполняем трансформацию
            # Преобразование одной точки
            lons, lats = transform(src.crs, {'init': 'epsg:4326'}, [xs], [ys])
            return lats[0], lons[0]

    except Exception as e:
        # print(f"Ошибка получения координат: {e}")
        return None, None


def get_address_from_coords(lat, lon):
    """
    Получение адреса по широте и долготе с использованием Nominatim.
    """
    try:
        geolocator = Nominatim(user_agent="geo_app")
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location:
            return location.address
        return "Неизвестное местоположение"
    except Exception:
        return "Ошибка поиска адреса"


def pixels_to_coords(tiff_path, pixel_points):
    """
    Преобразование списка точек (x, y) в координаты (lat, lon).
    pixel_points: список списков или кортежей [[x1, y1], [x2, y2], ...]
    Возвращает список [[lon, lat], ...] для GeoJSON (Lon, Lat порядок!).
    """
    coords = []
    try:
        with rasterio.open(tiff_path) as src:
            for pt in pixel_points:
                x, y = pt[0], pt[1]
                xs, ys = src.xy(y, x)
                # Обработка CRS
                if not src.crs:
                    # Если CRS не определена, предполагаем WGS84 (EPSG:4326)
                    lon, lat = xs, ys
                else:
                    try:
                        crs_code = src.crs.to_epsg()
                        if crs_code == 4326:
                            lon, lat = xs, ys
                        else:
                            lons_pf, lats_pf = transform(
                                src.crs, {'init': 'epsg:4326'}, [xs], [ys])
                            lon, lat = lons_pf[0], lats_pf[0]
                    except:
                        # Fallback при ошибке (например, локальная CRS)
                        lon, lat = xs, ys
                coords.append([lon, lat])
    except Exception as e:
        print(f"Ошибка конвертации полигона: {e}")
        return []
    return coords


def get_bounds(tiff_path):
    """
    Получение ограничивающего прямоугольника (min_lon, min_lat, max_lon, max_lat)
    GeoTIFF файла в формате EPSG:4326.
    """
    try:
        with rasterio.open(tiff_path) as src:
            bounds = src.bounds
            if src.crs.to_epsg() != 4326:
                # Трансформация границ в 4326
                xs, ys = transform(src.crs, {'init': 'epsg:4326'},
                                   [bounds.left, bounds.right],
                                   [bounds.bottom, bounds.top])
                return min(xs), min(ys), max(xs), max(ys)
            else:
                return bounds.left, bounds.bottom, bounds.right, bounds.top
    except Exception as e:
        print(f"Ошибка получения границ: {e}")
        return None
