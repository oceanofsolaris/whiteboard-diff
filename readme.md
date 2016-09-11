# Goal
The goal is to have e.g. an RPi with a webcam watching a
white/black-board that makes the current and past state of the board
available.

To do this, we need to

 1. Acquire a good picture of the board, ignoring people standing in
 front of it, temporary problems with lighting conditions and other
 problems.
 2. Locate a white/blackboard in the picture.
 3. Determine the aspect ratio of this board.
 4. Turn the boards content into a rectangular, clean image;
 homogenizing the background and normalize colors.
 5. Compare to previous state and only save if there is enough difference.
 6. Export the current and previous state.
As of now, only the steps 2. and 3. are implemented.

# Image acquisition

*To be implemented.*

The easiest way to do this would probably be taking multiple images
and taking the median of them. In addition it would be a good idea to
have a 'lights-off' mode that detects when the lightning conditions
are too bad to get any reliable image.

# Whiteboard detection

After acquiring the image, we search for the borders of a
whiteboard. We do this by searching for edges in the picture (using a
canny edge detection) and then taking a pseudo-radon transform*.
Long straight edges will show up as local maxima in this radon
transform, which we are easily able to detect.

Once we found the long straight edges, we need to find a combination
of them that forms a plausible rectangle. We try possible combinations
of lines until we find a set that forms a closed rectangle. Then we do
some fine-tuning to make sure we accurately portray the borders of
this rectangle that is probably the target white/blackboard.

This algorithm for finding rectangles roughly follows the approach
from the [Zhang et
al. (Microsoft) paper](http://research.microsoft.com/en-us/um/people/zhang/WhiteboardIt/)
with some slight differences. Most of differences are due to me only
becoming aware of this paper after already writing most of this myself.


Ideas for the future:

 * Save previously known whiteboard position, use  it as first guess
   for new rectangle. If we use a fixed camera position, this should
   allow us to skip all of the above in most cases.
 * Use Zhang et al. technique for edge detection and pseudo-radon
   transform: By employing a Sobel Operator instead of the Canny
   filter, we get edge orientation for free. This can be used for the
   pseudo-radon transform.

* We don't perform a true radon transform since that is quite
  slow. Instead, we isolate straight lines (using the findContours function
  from opencv) after performing the edge. We project these straight
  lines into the 'radon' space to get a much more efficient of the
  radon transform that works reasonably well for b/w images with
  straight lines.

# Aspect ratio estimation

Since we do not know in advance what the size of the whiteboard is, we
need to estimate its aspect ratio (otherwise we would distort the
content in the next steps). The Microsoft paper has an analytic
technique for this, which I did find rather unreliable though. For now
we do a brute force search for the right ratio (it is rather fast
anyways) by fitting a camera matrix and a target whiteboard to the
corners observed in the image. This is reliable enough as of now.

Ideas for the future:

 * Save last seen aspect ratio, try out if it produces a good fit.

# Cleaning the image

*To be implemented*

We know the quadrilateral the white/blackboard occupies in the image
as well as its aspect ratio. To get a proper image from this, we first
need to transform the quadrilateral into a rectangle with the correct
aspect ratio. This should can e.g. be done using the camera matrix
recovered during aspect ratio estimation and some kind of texture
filtering for good interpolation.

Then we need tohomogenize the background color and brightness
(ideas: strongly blurring the image and then subtracting this blurred image
from the image to remove background color; finding a mode in color
distribution to detect background color etc.).

# Compare to previous state to detect changes

*To be implemented*

The main differences here will probably be to consistently find
changes without triggering false alarms. These false alarms could be
due to:

 * Slight change in image orientation, boundaries, aspect ratio, color
 normalization and background removal due to errors in the previous
 steps. This would lead different 'cleaned' images from practically
 identical source images. This can be minimized by reusing the
 previous settings in these steps if possible.
 * Image noise, especially in bad lightning conditions.
 * White balance of the camera and lightning changes might lead to
 differently looking 'cleaned images' even if nothing changed on a
 white/blackboard. While detecting changes, we should probably
 ignore color to minimize the influence of this.

# Export the current and previous state

For starters, a simple shared directory would probably be a good
idea. Later, we could think about exposing the state through a
web-server and maybe adding some nifty features.