## CPU Chunking

* *important:* we actually don't split on platform changes anymore - transfers are happening automatically in the wrappers
* try to avoid files for slices/chunks as much as possible:
  * adapt the `BlockSplitter` and `BlockAggregator` to be able to go from/to file
  * should read/write chunks from memory if they can at all 
  * CPU memory estimation is highly variable - best is to try, and catch allocation errors - then react by swapping out to file (trial and error)
  * If allocation fails or it doesn't fit in memory anymore, the aggregator starts using files -> then file-based reslice is needed
  
* **GOAL:** use the same chunking loop as we have for GPUs also for CPUs

### Special Cases / Considerations

#### Loader

* Make sure that we can ask the loader for a block only, not the whole chunk
  (iteratively load)
* So we must include the loader in the blockwise loop
* "It behaves like the `BlockSplitter` when there's not enough memory for a chunk"

#### Reslice

* really needs be between sections
* needs to ask the block aggregator if it had to use files (didn't fit in memory)
* if yes, use file-based reslice
* if no, use in-memory reslice
* Idea:
  * what if the file-based `BlockAggregator`/`BlockSplitter` supported writing in one direction and reading in another?
  * this makes it not a "thing you insert between section" -> it's part of the sectioning

#### Global Stats

* methods needing them are the first in a section
* needs to be able to work cumulatively, block by block -> per-chunk stats
* then use MPI to get get global stats
* No need to start / stop a new section because of this
* could this be a block-wise method wrapper (that keeps state)?
  * accumulate the stats in the wrapper (state)
  * on the last block, it produces side outputs with the stats values
  * other methods can reference these side outputs...

#### Centering Meta-Wrapper

* It's method wrapper working block-by-block, producing a side output at the end of a chunk
* just like global stats

#### Save Intermediate Files

* Causes a split and starts a new section at the moment
* Must work block-by-block - cannot assume we can fit a whole chunk at once potentially
* Iterative appending to the files is necessary
* Options:
  * "It behaves like the `BlockAggregator` when there's not enough memory for a chunk"
  * Or: Implement like a method wrapper:
     *  --> No need to start / stop a new section because of this

#### Save to Images

* must work block-by block
* it's already a method wrapper - would work pretty much like saving intermediate files with a method wrapper


## Checkpointing (for future)

* Can be implemented by artifically inserting new section splits, and explicitly forcing the splitter/aggregator to go through files
* If block-wise state should be saved, need to be careful with methods that keep state:
  * perhaps have a method on the wrappers `get_state` and `set_state` that is rather abstract (any kind of state can be retrieved and set)
  * on a checkpoint (e.g. every 10sec), task runner would ask all methods in a section for their internal state and store that to file as well
  * When recovering, the last state at the checkpoint can be loaded and restored