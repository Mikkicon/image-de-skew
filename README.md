

TASK - Create a process for turning documents into book orientation
- Using an ML model that will __determine the angle from -30 to 30 degrees__.
- Libraries: __pytorch__ and tf and any supporting libraries.
- Time constraint: the model should run for __<5 seconds__, but this is not a critical requirement.
- Document the process sufficiently so it's and understandable during the review.
- requirements.txt file, which lists all the libraries used in the process with their versions
- deadline __11.09.2023__

## Usage
1. `docker compose up -d`
2. `docker exec -it deskew bash`
3. `python src/train.py`
4. `python src/deskew.py`

Check ./output directory for results