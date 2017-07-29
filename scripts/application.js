
/* This code is developed by PAN Hao based on previous code by spacekid 
 * whose source code can be found at https://github.com/ispacekid/color-tunes
 *
 * License under the MIT license
 */


$(document).ready(function() {
  $(".info").addClass("closed hidden transition transition-height");

  // function called when the poster is clicked
  $(".poster-picker").on("click", "a", function(event) {
 
    var movieSelected, posterActive, posterAnchor, posterSelected, image, indicatorPosition, infoHeight, switchInfoToMovie, toggleInfoForMovie,
      _this = this;
    posterAnchor = this;
    posterSelected = posterAnchor.parentNode;
    posterActive = $(".poster-picker .active");
    movieSelected = posterAnchor.href.split('#')[1];
    indicatorPosition = posterSelected.offsetLeft + (posterSelected.offsetWidth / 2) - 15;
    infoHeight = $(".info-inner")[0].getBoundingClientRect().height;
    
    function loadMovieInfo(movie) {
      var selectedMovie = MOVIE_INFO[movie];
      $('.movie-details').empty();
      
      for(var i = 0; i < selectedMovie.details.length; i++) {
        var detail = selectedMovie.details[i],
            detailListItem = $('<li>'),
            detailTitleItem = $('<h5>').addClass('detail-title').text(detail.title),
            detailContentItem = $('<span>').addClass('detail-content').text(detail.content);

        detailListItem.append(detailTitleItem).append(detailContentItem);
        detailListItem.appendTo('.movie-details');

      }

      // set the path of the poster for the image 6, which will be changed if a image is uploaded
      if (movieSelected == 6) {
        selectedMovie.poster = document.getElementById('tochange').src;
      }

      $('.movie-artwork').attr("src", selectedMovie.poster);
      $('.movie-title').text(selectedMovie.title);
      $('.movie-artist .artist-name').text(selectedMovie.artist);
      $('.movie-artist .released-on').text("• " + selectedMovie.release);
      $(".info").height(380);

      (function launchColorThiefLibrary(){
        var image = new Image();
        image.src = selectedMovie.poster;

        showColorsForImage(image);

      })();

    }; // end of function loadMovieInfo

    rgbToCssString = function(color) {
        return "rgb(" + color[0] + ", " + color[1] + ", " + color[2] + ")";
      };
    
    function showColorsForImage(image){

      var colorThief = new ColorThief();
      var bgColor      = colorThief.getColor(image);
      var palette    = colorThief.getPalette(image);

      generateColorBlocks();

        // get distanct of two point in color space
      colorDist = function(a, b) {
          var square;
          square = function(n) {
            return n * n;
          };
          return square(a[0] - b[0]) + square(a[1] - b[1]) + square(a[2] - b[2]);
        }; 


      // find the color from Palette that has largest dist to bg color
      maxDist = 0;
      for (_i = 0, _len = palette.length; _i < _len; _i++) {
        color = palette[_i];
        dist = colorDist(bgColor, color);
        if (dist > maxDist) {
          maxDist = dist;
          fgColor = color;
        }
      }

      // find another color from Palette that has 2nd largest dist to bg color
      maxDist = 0;
      for (_j = 0, _len1 = palette.length; _j < _len1; _j++) {
        color = palette[_j];
        dist = colorDist(bgColor, color);
        if (dist > maxDist && color !== fgColor) {
          maxDist = dist;
          fgColor2 = color;
        }
      }

      // Function for generating color blocks on the webpage; 
      // This part is completedly developed by PAN Hao
      function generateColorBlocks() {
        $('.palette').html('');

        $('.palette').append('<h4 class="text-muted">Dominant Color</h4>');

        var c = rgbToCssString(bgColor);
        //var newDiv = '<div class = "color-blocks"><div class="color-block" style="background-color: ' + c + '"></div></div>';
        $('.palette').append('<div class = "color-blocks"><div class="color-block" style="background-color: ' + c + '"></div></div>');

        $('.palette').append('<h4 class="text-muted">Palette</h4>');

        for (var i = 0; i < palette.length; i++) {
          var c = rgbToCssString(palette[i]);
          var newDiv = '<div class="color-block" style="background-color: ' + c + '"></div>';

          $('.palette').append(newDiv);
        }

      }

      // set the colors based on the color extracted
      $(".info").css("background-color", "" + (rgbToCssString(bgColor)));
      $(".info-indicator").css("border-bottom-color", "" + (rgbToCssString(bgColor)));
      $(".movie-title, .movie-details, .detail-title").css("color", "" + (rgbToCssString(fgColor)));
      $(".movie-divider").css("border-color", "" + (rgbToCssString(fgColor)));
      $(".movie-details, .detail-playtime").css("color", "" + (rgbToCssString(fgColor2)));
      return $(".movie-artist").css("color", "" + (rgbToCssString(fgColor2)));

    }

    toggleInfoForMovie = function(movie) {
      
      var isExpanding, targetHeight;
      isExpanding = $(".info").hasClass("closed");
      targetHeight = isExpanding ? $(".info").height : 0;

      if (isExpanding) {

        // If info expanded, hide the info
        $(".info").removeClass("hidden"); // show the info
        $(".info-indicator").removeClass("hidden").css("left", indicatorPosition); // show the info-indicator (triangle)

        $(".palette").removeClass("hidden");


      } else {

        // Else, show the info
        $(".info").on("webkitTransitionEnd", function() {
          $(".palette").addClass("hidden");
          $(".info-indicator").addClass("hidden"); // hide the info
          return $(".info").addClass("hidden").off("webkitTransitionEnd");
        });
      }
      $(posterSelected).toggleClass("active");
      return $(".info").toggleClass("closed expanded").height(targetHeight);
    }; //end of function toggleInfoForMovie


    switchInfoToMovie = function(movie) {
      $(".palette").removeClass("hidden");
      $(posterActive).removeClass("active");
      $(posterSelected).addClass("active"); // set the status of the selected poster as active
      return $(".info-indicator").css("left", indicatorPosition);
    }; // end of function switchInfoToMovie

    // call the function to get the Movie info
    loadMovieInfo(movieSelected); 

    if (($(posterActive).length === 0) || (posterSelected === posterActive[0])) {
      // when first click or click the same posters
      toggleInfoForMovie(movieSelected);
    } else if ($(posterActive).length > 0) {
      // when switch the posters (i.e. click on the different posters) 
      switchInfoToMovie(movieSelected);
    }

    posterAnchor.blur();
    event.preventDefault();
  
  });

  // Function for uploading image; This part is developed by PAN Hao
  $("#imgUpload").change(function(){
    // console.log('image uploaded');
    readURL(this);

    function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        
        reader.onload = function (e) {
            $('#tochange').attr('src', e.target.result);
        }
        
        reader.readAsDataURL(input.files[0]);
      }
    }
  });
   


});




