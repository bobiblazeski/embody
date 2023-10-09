# embody


Add transforms for dataset
 1 <del> Decimation [n::2, n::2] <del> 
 2 <del>Perlin noise (Upsample & add noise then decimate or downsamples)<del>
 3 <del> Random scale<del> 
 4 <del> Random shift <del> 
 5 <del> Random split <del> 
 6 <del> Reflection via padding <del> 
 7 <del> Random blend with other sample<del>
  
 ????

<del>Add tensorboard logging<del>
<del>Add checkpoints<del>
<del> Speed up training<del>
<del> Add samples logging<del>

<del> coarse resolution >>> vqgan = fine offsets<del> 
<del> coarse resolution + fine offsets = fine resolution<del> 
<del> Train with chamfer loss later<del>

<del>1. Use lexicon instead of codebook<del>
<del>2. Use categorical instead of argmax (softmax first)<del>
<del>3. Use epsilon greedy with argmax<del>
<del>4. Generate surface reoslution by resolution<del>
<del>5. Each resolution will have separate lexicon<del>
<del>6. Use inputs from previous resolution to get indices<del>
<del>7. Use skip to combine results<del>
<del>8. Use auxilary results to compare via chamfer loss<del>
<del>9. Use depthwise separable convolution<del>

## Tomorrow 
1. Fit single layer data

Train refiner

 
  ---Maybe simplify stl mesh with pyvtk ??



Maybe add weights slowly blending FMA

## Later:
### Diffuser (Generator)
Algorithm for training
 1 start with random codes
 2 denoise 
 3 add less noise 
 4 repeat 2 until last time step 

 How many steps 8? 20? or  8x8 for image size of 64 
 Try with smaller first

 Use temperature as paella


### Image encoder
1 Prepare datase of known face models and stls
2 Use adversarial loss from discriminator 
  -- Maybe train discriminator on codebook ????
3 Use face encoding loss between stl render and image encoding
