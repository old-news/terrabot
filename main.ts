import { fileLoader } from 'terraria-world-file/node'
import { FileReader, FileSaver, Position, TownNPC } from 'terraria-world-file'
import * as fs from 'node:fs'

async function run() {
const worldPath = '/home/drew/.local/share/Terraria/Worlds/The_Non-Fungible_Fountain.wld';
const parser = await new FileReader().loadFile(fileLoader, worldPath)
const world = parser.parse()

const spawnPosition: Position = {
  x: 16 * world.header.spawnTileX,
  y: 16 * world.header.spawnTileY
}

const wizard: TownNPC = {
  id: 108,
  name: 'Gandalf',
  position: spawnPosition,
  homePosition: spawnPosition,
  homeless: true,
  variationIndex: 0,
  homelessDespawn: false
}

//world.NPCs.townNPCs.push(wizard)
//world.NPCs.townNPCCount += 1
//console.log(JSON.stringify(world.NPCs.townNPCs[0]))
console.log(JSON.stringify(world.worldTiles.tiles[0]))

console.log(`Wizard has appeared at your base!`)

//const newFile = new FileSaver().save(world)
//fs.appendFileSync(worldPath, Buffer.from(newFile))
}

run();
