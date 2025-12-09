// =================== CHAT HISTORY UTILITIES (ESM) ===================
// Module-level dependencies - will be set by serverWml.js
let logger = console; // Default to console
let sock = null;
let chatStore = null;
let messageStore = null;
let connectionState = 'disconnected';
let formatJid = (jid) => jid;
let delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
let saveMessageToDB = () => {};
let performInitialSync = async () => {};
let saveAll = () => {};

// Initialize dependencies from main server
function initializeDependencies(deps) {
  logger = deps.logger || console;
  sock = deps.sock;
  chatStore = deps.chatStore;
  messageStore = deps.messageStore;
  connectionState = deps.connectionState;
  formatJid = deps.formatJid;
  delay = deps.delay;
  saveMessageToDB = deps.saveMessageToDB;
  performInitialSync = deps.performInitialSync;
  saveAll = deps.saveAll || (() => {});
}

// Getter for dynamic sock access
function getSock() {
  return sock;
}

function getConnectionState() {
  return connectionState;
}

async function loadChatHistory(jid, batchSize = Infinity) {
  const currentSock = getSock();
  if (!currentSock) {
    logger.warn('Cannot load chat history: socket not available');
    return [];
  }

  try {
    logger.info(`Loading FULL chat history for ${jid} (unlimited mode)`);

    const formattedJid = formatJid(jid);

    // Check if the method exists in Baileys
    if (!currentSock.fetchMessagesFromWA && !currentSock.loadMessages) {
      logger.warn('Chat history loading not available in this Baileys version');
      return [];
    }

    let cursor = undefined;
    let keepFetching = true;
    let allMessages = [];
    let fetchCount = 0;

    // UNLIMITED MODE - Continue until no more messages
    while (keepFetching) {
      let batch;

      // Fetch in optimal chunks (50 messages per request)
      const chunkSize = 50;

      // Try the available method
      if (currentSock.loadMessages) {
        batch = await currentSock.loadMessages(formattedJid, chunkSize, cursor);
      } else if (currentSock.fetchMessagesFromWA) {
        batch = await currentSock.fetchMessagesFromWA(formattedJid, chunkSize, cursor);
      }

      if (!batch || batch.length === 0) {
        logger.info(`No more messages for ${jid}, stopping at ${allMessages.length} messages`);
        break;
      }

      for (const msg of batch) {
        if (!msg.key?.id) continue;
        saveMessageToDB(msg, formattedJid);
        allMessages.push(msg);
      }

      cursor = batch[0]?.key;
      fetchCount++;

      // Progress logging every 10 batches (500 messages)
      if (fetchCount % 10 === 0) {
        logger.info(`Progress: ${allMessages.length} messages loaded for ${jid}...`);
      }

      // Small delay to avoid rate limiting
      await delay(200);
    }

    logger.info(`✓ Finished loading FULL history for ${jid}: ${allMessages.length} messages in ${fetchCount} batches`);
    return allMessages;
  } catch (err) {
    logger.error(`Failed to load chat history for ${jid}:`, err.message);
    return [];
  }
}




// =================== FUNZIONI DI SUPPORTO ===================

async function loadAllChatsHistory(maxChatsToLoad = Infinity, messagesPerChat = Infinity) {
  const currentSock = getSock();
  const currentState = getConnectionState();

  if (!currentSock || currentState !== 'open') {
    logger.warn('Cannot load all chats history: not connected');
    return false;
  }

  try {
    if (!chatStore || !chatStore.keys) {
      logger.warn('chatStore not available');
      return false;
    }

    const allChatIds = Array.from(chatStore.keys());
    const totalChats = allChatIds.length;

    logger.info(`═══════════════════════════════════════════════════`);
    logger.info(`🚀 FULL UNLIMITED SYNC STARTING`);
    logger.info(`   Total chats to process: ${totalChats}`);
    logger.info(`   Messages per chat: UNLIMITED`);
    logger.info(`   This will download EVERYTHING!`);
    logger.info(`═══════════════════════════════════════════════════`);

    let successCount = 0;
    let failCount = 0;
    let totalMessagesLoaded = 0;
    let processedChats = 0;

    for (const chatId of allChatIds) {
      try {
        processedChats++;
        logger.info(`\n[${processedChats}/${totalChats}] Processing chat: ${chatId}`);

        const messages = await loadChatHistory(chatId, messagesPerChat);
        if (messages.length > 0) {
          successCount++;
          totalMessagesLoaded += messages.length;
          logger.info(`✓ Chat ${chatId}: ${messages.length} messages loaded`);

          // Save progress every 10 chats
          if (successCount % 10 === 0) {
            logger.info(`\n💾 Saving progress... (${successCount} chats completed)`);
            saveAll();
          }
        } else {
          logger.info(`⚠ Chat ${chatId}: No messages found`);
        }

        // Delay between chats to avoid rate limiting
        await delay(500);

      } catch (chatError) {
        failCount++;
        logger.error(`✗ Failed to load history for ${chatId}:`, chatError.message);
        // Continue with next chat even if one fails
      }
    }

    logger.info(`\n═══════════════════════════════════════════════════`);
    logger.info(`✓ FULL UNLIMITED SYNC COMPLETED!`);
    logger.info(`   Total chats processed: ${processedChats}`);
    logger.info(`   Successful: ${successCount}`);
    logger.info(`   Failed: ${failCount}`);
    logger.info(`   Total messages loaded: ${totalMessagesLoaded}`);
    logger.info(`═══════════════════════════════════════════════════`);

    return successCount > 0;

  } catch (error) {
    logger.error('Bulk chat history load failed:', error.message);
    return false;
  }
}

async function loadRecentMessages(jid, hours = 24) {
  const cutoffTime = Math.floor((Date.now() - (hours * 60 * 60 * 1000)) / 1000);

  try {
    // Load ALL messages (unlimited)
    const messages = await loadChatHistory(jid, Infinity);

    logger.info(`Found ${messages.length} messages for ${jid}`);
    return messages;

  } catch (error) {
    logger.error(`Failed to load recent messages for ${jid}:`, error.message);
    return [];
  }
}

async function preloadImportantChats() {
  const currentSock = getSock();
  if (!currentSock) return;

  try {
    logger.info('Preloading ALL important chats (unlimited mode)...');

    if (!chatStore || !chatStore.entries) {
      logger.warn('chatStore not available for preloading');
      return;
    }

    // Identifica TUTTE le chat con attività recente (senza limite)
    const importantChats = Array.from(chatStore.entries())
      .map(([jid, messages]) => ({
        jid,
        messageCount: messages.length,
        lastActivity: messages.length > 0 ?
          Math.max(...messages.map(m => Number(m.messageTimestamp))) : 0
      }))
      .filter(chat => chat.lastActivity > 0) // Solo chat con messaggi
      .sort((a, b) => b.lastActivity - a.lastActivity);
      // NO LIMIT - Process ALL chats!

    logger.info(`Found ${importantChats.length} chats with activity`);

    for (const chat of importantChats) {
      try {
        // Load ALL messages for each chat (unlimited)
        await loadChatHistory(chat.jid, Infinity);
        await delay(500);
      } catch (error) {
        logger.warn(`Failed to preload chat ${chat.jid}:`, error.message);
      }
    }

    logger.info(`✓ Preloaded ${importantChats.length} important chats`);

  } catch (error) {
    logger.error('Important chats preload failed:', error.message);
  }
}

// Integrazione nel sistema esistente
async function enhancedInitialSync() {
  try {
    logger.info('\n');
    logger.info('╔═══════════════════════════════════════════════════════════╗');
    logger.info('║  ENHANCED UNLIMITED SYNC - DOWNLOADING EVERYTHING!       ║');
    logger.info('╚═══════════════════════════════════════════════════════════╝');
    logger.info('');

    // Prima esegui la sync base (contatti e lista chat)
    if (performInitialSync) {
      logger.info('📋 Step 1/2: Base sync (contacts & chat list)...');
      await performInitialSync();
      logger.info('✓ Base sync completed');
    }

    // Poi carica TUTTA la cronologia dei messaggi - NESSUN LIMITE!
    if (chatStore && chatStore.size > 0) {
      logger.info('\n📥 Step 2/2: Loading FULL chat histories (UNLIMITED MODE)...');
      logger.info(`   Found ${chatStore.size} chats to process`);
      logger.info('   This will download ALL messages from ALL chats!');
      logger.info('   Depending on your data, this may take a while...\n');

      // UNLIMITED SYNC - Scarica TUTTO!
      // No limits on chats, no limits on messages
      const success = await loadAllChatsHistory(Infinity, Infinity);

      if (success) {
        logger.info('\n✓ Enhanced sync completed successfully - ALL DATA DOWNLOADED!');
      } else {
        logger.warn('\n⚠ Enhanced sync completed with warnings');
      }

      // Final save of all data
      logger.info('\n💾 Saving all synchronized data to disk...');
      saveAll();
      logger.info('✓ All data saved successfully!');
    } else {
      logger.warn('⚠ No chats found to sync');
    }

    logger.info('\n');
    logger.info('╔═══════════════════════════════════════════════════════════╗');
    logger.info('║  SYNC COMPLETE - ALL YOUR DATA IS NOW AVAILABLE!        ║');
    logger.info('╚═══════════════════════════════════════════════════════════╝');
    logger.info('');

  } catch (error) {
    logger.error('\n✗ Enhanced sync failed:', error.message);
    logger.error(error.stack);
  }
}

// Export functions (ESM)
export {
  initializeDependencies,
  getSock,
  getConnectionState,
  loadChatHistory,
  loadAllChatsHistory,
  loadRecentMessages,
  preloadImportantChats,
  enhancedInitialSync
};
