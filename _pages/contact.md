---
title: "Contact"
toc: false
permalink: /contact/
read_time: false
share: false
related: false
comments: false
---


<html>
/* ==========================================================================
#CONTACT PAGE
========================================================================== */
/* an adaptation of http://tympanus.net/codrops/2015/01/08/inspiration-text-input-effects/ */

#contactform, .form {
    margin-top: 2rem;
}
.field {
    border: none;
    position: relative;
    z-index: 1;
    display: inline-block;
    margin: 0 1em 0 0;
    /* this is almost a third, on my own site I use calc(33% - 0.66rem) for width */
    width: 30%;
    vertical-align: top;
    overflow: hidden;
    font-family: "adelle-sans", "HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
}
.field:nth-child(3) {
    margin-right: 0;
}
.input {
    position: relative;
    display: block;
    float: right;
    border: none;
    border-radius: 0;
    -webkit-appearance: none; /* for box shadows to show on iOS */
    width: 100%;
    background: transparent;
    padding: 0.5em;
    margin-bottom: 2em;
    z-index: 100;
    opacity: 0;
    height: 2rem;
}

textarea.input {
    resize: none;
    padding-bottom: 0;
}

.input:focus {
    outline: none;
}

.label {
    display: inline-block;
    float: right;
    color: hsla(221,72%,55%,1);
    font-weight: bold;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    -webkit-touch-callout: none;
    -webkit-user-select: none;
    -khtml-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
    width: 100%;
    position: absolute;
    text-align: left;
    padding: 0.5em 0;
    pointer-events: none;
    font-size: 1em;
}
.label::before,
.label::after {
    content: '';
    position: absolute;
    width: 100%;
    left: 0;
}
.label::before {
    height: 100%;
    background: #fff;
    top: 0;
    -webkit-transform: translate3d(0, -100%, 0);
    transform: translate3d(0, -100%, 0);
    -webkit-transition: -webkit-transform 0.3s;
    transition: transform 0.3s;
}
.label::after {
    height: 2px;
    background: hsla(221,72%,55%,1);
    top: 100%;
    -webkit-transition: opacity 0.3s;
    transition: opacity 0.3s;
}

.label-content {
    position: relative;
    display: block;
    width: 100%;
    padding: 0;
    -webkit-transform-origin: 0 0;
    transform-origin: 0 0;
    -webkit-transition: -webkit-transform 0.3s, color 0.3s;
    transition: transform 0.3s, color 0.3s;
}

.input:focus,
.input--filled .input {
    opacity: 1;
    -webkit-transition: opacity 0s 0.3s;
    transition: opacity 0s 0.3s;
}

.input:focus + .label::before,
.input--filled .label::before {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
}

.input:focus + .label::after,
.input--filled .label::after {
    opacity: 0;
}

.input:focus + .label .label-content,
.input--filled .label .label-content {
    color: #cbc4c6;
    -webkit-transform: translate3d(0, 2.1em, 0) scale3d(0.65, 0.65, 1);
    transform: translate3d(0, 2.1em, 0) scale3d(0.65, 0.65, 1);
}

.button, input[type=submit] {
    -webkit-appearance: none;
    border: none;
    border-radius: 0.25rem;
    padding: 0.5em;
    color: #fff;
    background-color: hsla(221,72%,55%,1);
    transition: all 0.3s ease-out;
    -webkit-box-shadow: 0 2px 5px 0 rgba(0,0,0,0.18),0 2px 10px 0 rgba(0,0,0,0.15);
    -moz-box-shadow: 0 2px 5px 0 rgba(0,0,0,0.18),0 2px 10px 0 rgba(0,0,0,0.15);
    box-shadow: 0 2px 5px 0 rgba(0,0,0,0.18),0 2px 10px 0 rgba(0,0,0,0.15);
}
.button:hover, .button:focus, .button:active {
    -webkit-box-shadow: 0 5px 11px 0 rgba(0,0,0,0.23),0 4px 15px 0 rgba(0,0,0,0.20);
    -moz-box-shadow: 0 5px 11px 0 rgba(0,0,0,0.23),0 4px 15px 0 rgba(0,0,0,0.20);
    box-shadow: 0 5px 11px 0 rgba(0,0,0,0.23),0 4px 15px 0 rgba(0,0,0,0.20);
}

/*!
 * classie v1.0.1
 * class helper functions
 * from bonzo https://github.com/ded/bonzo
 * MIT license
 *
 * classie.has( elem, 'my-class' ) -> true/false
 * classie.add( elem, 'my-new-class' )
 * classie.remove( elem, 'my-unwanted-class' )
 * classie.toggle( elem, 'my-class' )
 */

/*jshint browser: true, strict: true, undef: true, unused: true */
/*global define: false, module: false */

( function( window ) {

'use strict';

// class helper functions from bonzo https://github.com/ded/bonzo

function classReg( className ) {
  return new RegExp("(^|\\s+)" + className + "(\\s+|$)");
}

// classList support for class management
// altho to be fair, the api sucks because it won't accept multiple classes at once
var hasClass, addClass, removeClass;

if ( 'classList' in document.documentElement ) {
  hasClass = function( elem, c ) {
    return elem.classList.contains( c );
  };
  addClass = function( elem, c ) {
    elem.classList.add( c );
  };
  removeClass = function( elem, c ) {
    elem.classList.remove( c );
  };
}
else {
  hasClass = function( elem, c ) {
    return classReg( c ).test( elem.className );
  };
  addClass = function( elem, c ) {
    if ( !hasClass( elem, c ) ) {
      elem.className = elem.className + ' ' + c;
    }
  };
  removeClass = function( elem, c ) {
    elem.className = elem.className.replace( classReg( c ), ' ' );
  };
}

function toggleClass( elem, c ) {
  var fn = hasClass( elem, c ) ? removeClass : addClass;
  fn( elem, c );
}

var classie = {
  // full names
  hasClass: hasClass,
  addClass: addClass,
  removeClass: removeClass,
  toggleClass: toggleClass,
  // short names
  has: hasClass,
  add: addClass,
  remove: removeClass,
  toggle: toggleClass
};

// transport
if ( typeof define === 'function' && define.amd ) {
  // AMD
  define( classie );
} else if ( typeof exports === 'object' ) {
  // CommonJS
  module.exports = classie;
} else {
  // browser global
  window.classie = classie;
}

})( window );

[].slice.call( document.querySelectorAll( '.input' ) ).forEach( function( inputEl ) {
    // in case the input is already filled..
    if( inputEl.value.trim() !== '' ) {
        classie.add( inputEl.parentNode, 'input--filled' );
    }

    // events:
    inputEl.addEventListener( 'focus', onInputFocus );
    inputEl.addEventListener( 'blur', onInputBlur );
} );

function onInputFocus( ev ) {
      classie.add( ev.target.parentNode, 'input--filled' );
}

function onInputBlur( ev ) {
      if( ev.target.value.trim() === '' ) {
            classie.remove( ev.target.parentNode, 'input--filled' );
      }
}


<form class="form" id="contactform" action="//formspree.io/SarfarazAAbbasi+TheSarfarazGitHub@Gmail.Com.com" method="POST">
 <fieldset class="field">
 <input class="input" type="text" name="name" placeholder="Name" id="name" required>
 <label class="label" for="name"><span class="label-content">Your name</span></label>
 </fieldset>
 <fieldset class="field">
 <input class="input" type="email" name="_replyto" placeholder="example@domain.com" id="_replyto" required>
 <label class="label" for="_replyto"><span class="label-content">Your email</span></label>
 </fieldset>
 <fieldset class="field">
 <textarea class="input" name="message" rows="1" placeholder="Message" id="message" required></textarea>
 <label class="label" for="message"><span class="label-content">Your message</span></label>
 </fieldset>
 <input class="hidden" type="text" name="_gotcha" style="display:none">
 <input class="hidden" type="hidden" name="_subject" value="Message via http://domain.com">
 <fieldset class="field">
 <input class="button submit" type="submit" value="Send">
 </fieldset>
</form>


</html>
