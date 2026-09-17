import os
import re
from datetime import datetime
from pymongo import MongoClient

# ── MongoDB Configuration ─────────────────────────────────────────────────────
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "")
MONGO_DATABASE = 'arcis'
# Overridable so a run can be isolated from unrelated data already in the
# database: arcis.activities1 holds 1048 segments from a previous indoor-camera
# deployment (Reception, Canteen, B1-STairs...) which would otherwise be
# returned by every search alongside this dataset's footage.
#
# Deliberately NOT named MONGO_COLLECTION_ACTIVITIES: the checked-in .env sets
# that to 'activities' (and MONGO_DATABASE to 'arcis-railway'), which is stale -
# arcis-railway.activities holds 33 documents while arcis.activities1 holds 1048.
# Reading that variable would silently repoint the whole system at the wrong
# collection for anything that calls load_dotenv().
MONGO_COLLECTION_ACTIVITIES = os.getenv('ACTIVITIES_COLLECTION', 'activities1')

# Initialize MongoDB client
client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=30000)
db_mongo = client[MONGO_DATABASE]
collection = db_mongo[MONGO_COLLECTION_ACTIVITIES]

# The embedding is 768 floats (~15 KB of JSON) per segment and is never used by
# the UI - only by semantic search inside the server. Excluding it is the single
# biggest win in page load time.
_SEGMENT_PROJECTION = {"embedding": 0}


def load_segments_page(skip: int = 0, limit: int = 5, camera: str = "",
                       date: str = "", query: str = ""):
    """One page of segments, filtered and paginated in MongoDB.

    Filtering server-side matters as much as paginating: the page used to pull
    every segment down and filter in the browser, so the cost grew with the
    whole corpus rather than with what was being shown.
    """
    q = {}
    if camera and camera != "All":
        q["camera_id"] = camera
    if date and date != "All":
        q["$expr"] = {"$eq": [
            {"$substr": [{"$toString": "$start_time"}, 0, 10]}, date]}
    if query:
        q["description"] = {"$regex": re.escape(query), "$options": "i"}

    total = collection.count_documents(q)
    docs = list(collection.find(q, _SEGMENT_PROJECTION)
                .sort("start_time", -1).skip(max(0, skip)).limit(max(1, limit)))
    return _shape(docs), total


def segment_facets():
    """Distinct cameras and dates, for the filter dropdowns."""
    cameras = sorted(c for c in collection.distinct("camera_id") if c)
    dates = set()
    for t in collection.distinct("start_time"):
        if isinstance(t, datetime):
            dates.add(t.strftime("%Y-%m-%d"))
        elif t:
            dates.add(str(t)[:10])
    return cameras, sorted(dates, reverse=True)


def _shape(docs):
    """Normalise raw documents for the UI (dates as strings, id as text)."""
    out = []
    for seg_dict in docs:
        if "_id" in seg_dict:
            seg_dict["_id"] = str(seg_dict["_id"])
        seg_dict.setdefault("camera_id", "Unknown")
        for key in ("start_time", "end_time"):
            val = seg_dict.get(key)
            seg_dict[f"{key}_str"] = (val.strftime('%Y-%m-%d %H:%M:%S')
                                      if isinstance(val, datetime) else str(val or ''))
        seg_dict.setdefault("description", "No description available")
        out.append(seg_dict)
    return out


def load_all_segments():
    """Load all segments from MongoDB. Always reads fresh — no cache."""
    all_segments = []
    try:
        # Fetch all documents from MongoDB collection
        docs = list(collection.find().sort("start_time", -1))
        
        for doc in docs:
            # Convert MongoDB doc to dict if needed (it already is one)
            seg_dict = doc
            # Remove _id as it's not JSON serializable easily
            if "_id" in seg_dict:
                seg_dict["_id"] = str(seg_dict["_id"])
            
            # Ensure camera_id is present
            if "camera_id" not in seg_dict:
                seg_dict["camera_id"] = "Unknown"

            # Handle start_time and end_time
            if isinstance(seg_dict.get('start_time'), datetime):
                seg_dict['start_time_str'] = seg_dict['start_time'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                seg_dict['start_time_str'] = str(seg_dict.get('start_time', ''))

            if isinstance(seg_dict.get('end_time'), datetime):
                seg_dict['end_time_str'] = seg_dict['end_time'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                seg_dict['end_time_str'] = str(seg_dict.get('end_time', ''))

            if 'description' not in seg_dict:
                seg_dict['description'] = "No description available"

            all_segments.append(seg_dict)
    except Exception as e:
        print(f"Failed to load segments from MongoDB: {e}")

    return all_segments


def delete_segments_from_mongo(to_delete: list):
    """
    Permanently remove selected segments from MongoDB.

    to_delete: list of (camera_id, segment_id) tuples.
    Returns (number_deleted, list_of_error_strings).
    """
    if not to_delete:
        return 0, []

    total_deleted = 0
    errors = []

    for item in to_delete:
        try:
            camera_id = item[0]
            segment_id = item[1]
            # Delete the specific segment from MongoDB
            result = collection.delete_one({
                "camera_id": camera_id,
                "segment_id": segment_id
            })
            total_deleted += result.deleted_count
        except Exception as e:
            errors.append(f"Error deleting segment for camera {item}: {e}")

    return total_deleted, errors
