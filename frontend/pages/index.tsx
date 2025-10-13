import Head from 'next/head'
import TownView from '../components/TownView'

export default function Home() {
  return (
    <>
      <Head>
        <title>Mini-Town Control Room</title>
        <meta name="description" content="Monitor Mini-Town agents and manage the simulation." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <TownView />
    </>
  )
}
